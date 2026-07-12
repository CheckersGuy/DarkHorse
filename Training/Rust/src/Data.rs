#![feature(iter_array_chunks)]
use crate::Move;
use crate::Pos::Position;
use crate::Pos::Square;
use crate::Sample;
use crate::Sample::Game;
use crate::Sample::OldSample;
use crate::Sample::SampleIteratorTrait;
use crate::TableBase;
use bloomfilter::reexports::bit_vec::BitBlock;
use bloomfilter::Bloom;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use libc::abs;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rip_shuffle::RipShuffleParallel;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::format;
use std::fs::File;
use std::fs::OpenOptions;
use std::hash::Hash;
use std::io::{BufRead, Write};
use std::io::{BufReader, BufWriter};
use std::ops::Div;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use Sample::{Result, SampleType};
#[derive(Debug)]
pub struct Generator<'a> {
    book: String,
    output: String,
    num_workers: usize,
    pub max_samples: usize,
    pub time: usize,
    pub max_nodes: usize,
    pub depth: usize,
    pub prev_file: Option<&'a str>,
}

const RESCORE_VALUE_THRESHOLD: i16 = 5000; // don't bother rescoring decisive/TB-range evals

pub fn rescoring_data(
    paths: Vec<&str>,
    output: &str,
    time: i32,
    num_workers: usize,
    partitions: usize,
    queue_capacity: usize,
) -> std::io::Result<()> {
    let mut total_count: u64 = 0;
    let mut skipped_rescoring: u64 = 0;

    let (work_tx, work_rx) = mpsc::sync_channel::<Sample::Sample>(queue_capacity);
    let work_rx = Arc::new(Mutex::new(work_rx));

    let (result_tx, result_rx) = mpsc::sync_channel::<Sample::Sample>(queue_capacity);

    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {spinner} produced: {msg}").unwrap(),
    );

    // --- writer thread ---
    let written_counter = Arc::new(AtomicUsize::new(0));
    let writer_written_counter = Arc::clone(&written_counter);
    let output_owned = output.to_string();
    let writer_handle = {
        let mut files: Vec<BufWriter<std::fs::File>> = Vec::new();
        for i in 0..partitions {
            let file_name = format!("{}{}", output_owned, i);
            files.push(BufWriter::new(File::create(file_name)?));
        }
        thread::spawn(move || {
            let mut rng = StdRng::from_rng(thread_rng()).unwrap();
            for sample in result_rx.iter() {
                let partition = rng.gen::<usize>() % partitions;
                sample
                    .write_fen(&mut files[partition])
                    .expect("Could not write rescored sample");
                writer_written_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        })
    };

    // --- worker threads ---
    let mut worker_handles = Vec::new();
    for worker_id in 0..num_workers {
        let work_rx = Arc::clone(&work_rx);
        let result_tx = result_tx.clone();
        let handle = thread::spawn(move || {
            let mut command = Command::new("./generator2")
                .args([format!("--eval-loop --time {} --hash_size 64", time)])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("Failed to start process");
            let mut stdin = command.stdin.take().unwrap();
            let stdout = command.stdout.take().unwrap();
            let mut f = BufReader::new(stdout);

            loop {
                let received = {
                    let rx = work_rx.lock().unwrap();
                    rx.recv()
                };
                let mut sample = match received {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let fen_string = sample.position.get_fen_string();
                if stdin.write_all((fen_string + "\n").as_bytes()).is_err() {
                    break;
                }

                let mut buffer = String::new();
                if let Err(e) = f.read_line(&mut buffer) {
                    println!("Worker {worker_id} read error: {:?}", e);
                    continue;
                }
                buffer = buffer.trim().replace("\n", "");
                let eval: i16 = match buffer.parse() {
                    Ok(v) => v,
                    Err(_) => {
                        println!("Worker {worker_id} could not parse eval '{buffer}'");
                        continue;
                    }
                };
                sample.value = eval;

                if result_tx.send(sample).is_err() {
                    break;
                }
            }

            let _ = stdin.write_all(b"terminate\n");
            let _ = command.wait();
            println!("Worker {worker_id} finished");
        });
        worker_handles.push(handle);
    }

    // --- main thread: producer ---
    let mut filter = Bloom::new_for_fp_rate(4_000_000_000, 0.01);
    for path in paths.iter() {
        println!("Reading games from file: {}", path);
        let mut reader = BufReader::new(File::open(path)?);
        for game in reader.iter_games() {
            for sample in game.get_samples() {
                if sample.position.has_capture() {
                    continue;
                }
                if sample.position.bp == 0 || sample.position.wp == 0 {
                    continue;
                }
                if sample.value.abs() >= 15000 {
                    continue;
                }
                if filter.check(&sample.position) {
                    continue;
                }
                filter.set(&sample.position);

                total_count += 1;
                bar.set_message(total_count.to_string());
                bar.tick();

                if sample.value.abs() >= RESCORE_VALUE_THRESHOLD {
                    // already decisive / TB-range: keep the old value, skip the engine call,
                    // but still route it through dedup+write like everything else
                    skipped_rescoring += 1;
                    if result_tx.send(sample).is_err() {
                        break;
                    }
                    continue;
                }

                if work_tx.send(sample).is_err() {
                    break;
                }
            }
        }
    }

    drop(work_tx);
    drop(result_tx); // drop main's own clone too, now that producing is done
    for handle in worker_handles {
        handle.join().unwrap();
    }
    writer_handle.join().unwrap();

    let written_count = written_counter.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "Rescored and wrote back {} unique samples out of {} candidates ({} skipped rescoring, kept old value)",
        written_count, total_count, skipped_rescoring
    );

    // --- shuffle + merge partitions ---
    let mut writer = BufWriter::new(File::create(output)?);
    let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    for i in 0..partitions {
        let file_name = format!("{}{}", output, i);
        let mut read_local = BufReader::new(File::open(&file_name)?);
        let mut samples: Vec<Sample::Sample> = read_local.iter_samples().collect();
        samples.par_shuffle(&mut rng);
        println!("Done shuffling partition {i}");
        for sample in samples {
            sample.write_fen(&mut writer)?;
        }
    }
    writer.flush()?;

    Ok(())
}

pub fn create_book(input: &str, output: &str, num_workers: usize) -> std::io::Result<()> {
    //create an opening book
    let (tx, rx): (Sender<String>, Receiver<String>) = mpsc::channel();
    let open_reader = BufReader::new(File::open(input)?);
    let mut writer = File::create(output)?;
    let openings: Vec<String> = open_reader.lines().map(|value| value.unwrap()).collect();
    let mut filter = Bloom::new_for_fp_rate(1000000000, 0.01);

    let bar = ProgressBar::new(openings.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    for chunk in openings.chunks(openings.len() / num_workers) {
        let sender = tx.clone();
        let my_chunk = chunk.to_owned();
        thread::spawn(move || {
            let mut command = Command::new("./MainEngine")
                .args(["--book"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("Failed to start process");
            let mut stdin = command.stdin.take().unwrap();
            let stdout = command.stdout.take().unwrap();
            let mut f = BufReader::new(stdout);

            for pos in my_chunk {
                stdin.write_all((pos.clone() + "\n").as_bytes()).unwrap();
                'generate: loop {
                    let mut buffer = String::new();
                    match f.read_line(&mut buffer) {
                        Ok(_) => {}
                        Err(e) => {
                            println!("{:?}", e)
                        }
                    }
                    buffer = buffer.trim().replace("\n", "");
                    sender.send(buffer.clone()).unwrap();
                    if buffer == "done" {
                        break 'generate;
                    }
                }
            }
            stdin
                .write_all((String::from("terminate\n")).as_bytes())
                .unwrap();

            command.kill().unwrap();
        });
    }
    let mut u_count: usize = 0;
    for val in rx {
        let trimmed = val.trim().replace("\n", "").to_lowercase();
        if trimmed != "done" && !filter.check(&val) {
            //println!("{val}");
            writer.write_all((val.clone() + "\n").as_bytes()).unwrap();
            u_count += 1;
            filter.set(&val);
        } else if trimmed == "done" {
            bar.inc(1);
        }
    }
    Ok(())
}

pub fn shuffle_data_external<const partitions: usize>(
    input: &str,
    output: &str,
) -> std::io::Result<()> {
    let mut files: Vec<BufWriter<std::fs::File>> = Vec::new();
    let mut writer = BufWriter::new(File::create(output)?);
    let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    for i in 0..partitions {
        let file_name = String::from(input) + i.to_string().as_str();
        files.push(BufWriter::new(File::create(file_name)?));
    }

    let mut reader = BufReader::new(File::open(input)?);

    //iterate over all samples

    for sample in reader.iter_samples() {
        //picking a random partition for our sample
        let partition = rand::thread_rng().gen::<usize>() % partitions;
        sample.write_fen(&mut files[partition])?;
    }

    println!("Done creating partitions");
    files.clear();
    for i in 0..partitions {
        let file_name = String::from(input) + i.to_string().as_str();
        let mut read_local = BufReader::new(File::open(file_name)?);
        let mut samples: Vec<Sample::Sample> = read_local.iter_samples().collect();
        samples.par_shuffle(&mut rng);
        println!("Done shuffling partition {i}");
        for sample in samples {
            sample.write_fen(&mut writer)?;
        }
    }

    Ok(())
}

//remove samples from a dataset
pub fn remove_samples(input: &str, removers: &str, output: &str) -> std::io::Result<()> {
    let mut filter = Bloom::new_for_fp_rate(30000000, 0.001);
    let mut writer = BufWriter::new(File::create(output)?);
    let mut counter = 0;
    let mut rem_counter = 0;
    {
        let mut reader = BufReader::new(File::open(removers)?);
        for sample in reader.iter_samples() {
            filter.set(&sample.position);
            counter += 1;
        }
    }
    let mut reader = BufReader::new(File::open(input)?);
    for sample in reader.iter_samples() {
        if !filter.check(&sample.position) {
            sample.write_fen(&mut writer)?;
        } else {
            rem_counter += 1;
        }
    }
    println!(
        "Removed {} of {} possible removable samples",
        rem_counter, counter
    );

    Ok(())
}

pub fn create_mlh_data(paths: Vec<&str>, output: &str) -> std::io::Result<()> {
    let mut writer = BufWriter::with_capacity(10000, File::create(output)?);
    let mut filter = Bloom::new_for_fp_rate(4000000000, 0.01);
    let mut total_count: u64 = 0;
    let mut written_count: u64 = 0;

    for path in paths {
        let mut reader = BufReader::with_capacity(1000000, File::open(path)?);
        for game in reader.iter_games() {
            let mut mlh_counter = 0;

            if game.result == Result::DRAW || game.result == Result::UNKNOWN {
                continue;
            }

            let samples = game.get_samples();
            for sample in samples.iter() {
                mlh_counter += 1;
                if sample.position.has_capture() {
                    continue;
                }
                if (sample.position.bp == 0) || (sample.position.wp == 0) {
                    continue;
                }
                if sample.value.abs() >= 15000 {
                    continue;
                }
                if sample.value.abs() <= 150 {
                    continue;
                }
                if !filter.check(&sample.position) {
                    filter.set(&sample.position);
                    let mut copy = sample.clone();
                    copy.mlh = mlh_counter;
                    copy.write_fen(&mut writer)?;
                    written_count += 1;
                }
                total_count += 1;
            }
        }
    }

    writer.flush()?;

    println!(
        "Got back a total of {} while processing {} samples",
        written_count, total_count
    );
    Ok(())
}

pub fn filter_training_data(path: &str, out: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = BufWriter::new(File::create(out)?);
    let sample_iter = reader.iter_samples();
    let mut filter = Bloom::new_for_fp_rate(5000000000, 0.05);
    let mut unique_counter = 0;
    let mut rescoreable_posititions = 0;

    //filtering positions where the material imbalance is too high
    //might be a good idea as well
    //
    for s in sample_iter {
        if s.value.abs() >= 650 && !s.is_tb_position() {
            continue;
        }

        let from_white_pov = s.position;

        let white_material = from_white_pov.wp.count_ones() as i32
            + (from_white_pov.k & from_white_pov.wp).count_ones() as i32;
        let black_material = from_white_pov.bp.count_ones() as i32
            + (from_white_pov.k & from_white_pov.bp).count_ones() as i32;
        let diff = white_material - black_material;
        if diff.abs() >= 2 {
            continue;
        }
        if !filter.check(&from_white_pov) {
            filter.set(&from_white_pov);
            s.write_fen(&mut writer)?;
            unique_counter += 1;
            rescoreable_posititions += match s.result {
                Result::TBWIN => 0,
                Result::TBLOSS => 0,
                Result::TBDRAW => 0,
                _ => 1,
            };
        }
    }
    println!("There are a total of {} positions in the new dataset. Of the {} positions, {} are rescored tablebase positions", unique_counter,unique_counter,rescoreable_posititions);
    Ok(())
}

pub fn get_unique_samples(
    paths: Vec<&str>,
    output: &str,
    partitions: usize,
    base: &TableBase::Base,
) -> std::io::Result<()> {
    //let mut reader = BufReader::new(File::open(path)?);
    let mut filter = Bloom::new_for_fp_rate(4000000000, 0.01);
    let mut total_count: u64 = 0;
    let mut written_count: u64 = 0;

    let mut files: Vec<BufWriter<std::fs::File>> = Vec::new();
    let mut writer = BufWriter::new(File::create(output)?);
    let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    for i in 0..partitions {
        let file_name = String::from(output) + i.to_string().as_str();
        files.push(BufWriter::new(File::create(file_name)?));
    }

    println!("Starting to write files");
    for path in paths {
        println!("Starting with file: {}", path);
        let mut reader = BufReader::new(File::open(path)?);
        for game in reader.iter_games() {
            let samples = game.get_samples();

            for sample in samples {
                if sample.position.has_capture() {
                    continue;
                }
                if (sample.position.bp == 0) || (sample.position.wp == 0) {
                    continue;
                }
                if sample.value.abs() >= 15000 {
                    continue;
                }

                if !filter.check(&sample.position) {
                    filter.set(&sample.position);

                    //checkinig if the position is in the tb

                    let tb_probe = base
                        .probe_with_position(sample.position)
                        .expect("Could not probe the tablebase");
                    let mut copy = sample.clone();
                    copy.value = match tb_probe {
                        Result::TBLOSS => -10000,
                        Result::TBWIN => 10000,
                        Result::TBDRAW => 0,
                        _ => sample.value,
                    };

                    let partition = rand::thread_rng().gen::<usize>() % partitions;
                    copy.write_fen(&mut files[partition])?;
                    written_count += 1;
                }
                total_count += 1;
            }
        }
    }
    //
    println!("Done sampling unique positions and creating partitions\n Now we are shuffling and merging the files");
    files.clear(); //that should flush the buffers as well
    for i in 0..partitions {
        let file_name = String::from(output) + i.to_string().as_str();
        let mut read_local = BufReader::new(File::open(file_name)?);
        let mut samples: Vec<Sample::Sample> = read_local.iter_samples().collect();
        samples.par_shuffle(&mut rng);
        println!("Done shuffling partition {i}");
        for sample in samples {
            sample.write_fen(&mut writer)?;
        }
    }

    writer.flush()?;
    println!(
        "Got back a total of {} while processing {} samples",
        written_count, total_count
    );
    Ok(())
}
pub fn create_policy_data(
    paths: Vec<&str>,
    output: &str,
    partitions: usize,
) -> std::io::Result<()> {
    let mut filter = Bloom::new_for_fp_rate(4000000000, 0.01);
    let mut written_count: usize = 0;
    let mut total_count: usize = 0;
    let mut files: Vec<BufWriter<std::fs::File>> = Vec::new();
    let mut writer = BufWriter::new(File::create(output)?);
    let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    for i in 0..partitions {
        let file_name = String::from(output) + i.to_string().as_str();
        files.push(BufWriter::new(File::create(file_name)?));
    }

    for path in paths {
        let mut reader = BufReader::new(File::open(path)?);
        println!("Starting reading file {}", path);
        for game in reader.iter_games() {
            let samples = game.get_samples();

            for window in samples.windows(2) {
                let next_pos = window[1].position.get_color_flip();
                let prev_pos = window[0].position;
                if prev_pos.has_capture() {
                    continue;
                }
                if (prev_pos.bp == 0) || (prev_pos.wp == 0) {
                    continue;
                }

                let move_encoding =
                    Move::get_move_encoding_from_pos(prev_pos, next_pos).unwrap_or(-1);

                if move_encoding >= 0 && !filter.check(&prev_pos) {
                    let mut sample = window[0].clone();
                    if sample.position.color == -1 {
                        sample.position = sample.position.get_color_flip();
                    }
                    sample.mlh = move_encoding as i16;
                    let partition = rand::thread_rng().gen::<usize>() % partitions;
                    written_count += 1;
                    sample.write_fen(&mut files[partition])?;
                    filter.set(&prev_pos);
                }
                total_count += 1;
            }
        }
    }

    println!("Done sampling unique positions and creating partitions\n Now we are shuffling and merging the files");
    for i in 0..partitions {
        let file_name = String::from(output) + i.to_string().as_str();
        let mut read_local = BufReader::new(File::open(file_name)?);
        let mut samples: Vec<Sample::Sample> = read_local.iter_samples().collect();
        samples.par_shuffle(&mut rng);
        println!("Done shuffling partition {i}");
        for sample in samples {
            sample.write_fen(&mut writer)?;
        }
    }

    writer.flush()?;
    println!(
        "Got back a total of {} while processing {} samples",
        written_count, total_count
    );

    Ok(())
}

pub fn shuffle_data(path: &str, output: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = BufWriter::new(File::create(output)?);
    let mut samples = Vec::new();

    for sample in reader.iter_samples() {
        samples.push(sample);
    }
    let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    samples.par_shuffle(&mut rng);

    for sample in samples {
        sample
            .write_fen(&mut writer)
            .expect("Error writing back data");
    }

    Ok(())
}

impl<'a> Generator<'a> {
    pub fn new(
        book: String,
        output: String,
        num_workers: usize,
        max_samples: usize,
    ) -> Generator<'a> {
        Generator {
            book,
            output: output,
            num_workers: num_workers,
            max_samples: max_samples,
            time: 10,
            depth: 128,
            max_nodes: 18446744073709551615usize,
            prev_file: None,
        }
    }

    pub fn generate_games(&self) -> std::io::Result<()> {
        let mut filter = Bloom::new_for_fp_rate(3000000000, 0.1);
        let mut unique_count = 0;
        let mut total_count = 0;
        let time = self.time;
        let max_nodes = self.max_nodes;
        let depth = self.depth;
        let mut writer = BufWriter::new(File::create(self.output.clone())?);
        let thread_counter = Arc::new(AtomicUsize::new(0));
        let opening_counter = Arc::new(Mutex::new(0));

        let mut handles = Vec::new();
        let reader = BufReader::with_capacity(1000000, File::open(self.book.clone())?);
        let openings = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx): (Sender<Vec<String>>, Receiver<Vec<String>>) = mpsc::channel();
        for line in reader.lines().skip(1) {
            {
                let result = line?;
                let mut guard = openings.lock().unwrap();
                guard.push(result.clone());
            }
        }

        let bar = ProgressBar::new(self.max_samples as u64);
        bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise},{per_sec}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),

    );

        let mut average_game_length: f32 = 0.0;
        let mut game_count: f32 = 0.0;
        let max_samples = self.max_samples;
        for id in 0..self.num_workers {
            let open = Arc::clone(&openings);
            let op_counter = Arc::clone(&opening_counter);
            let sender = tx.clone();
            let counter = Arc::clone(&thread_counter);
            let worker_seed: u64 =
                rand::thread_rng().gen::<u64>() ^ (id as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let handle = std::thread::spawn(move || {
                let mut command = Command::new("./MainEngine")
                    .args([format!(
                        "--generate --time {} --nodes {} --depth {} --seed {}
                         --adj_draw_count 8
                         --adj_draw_score 5
                         --adj_draw_min_ply 10
                         --adj_draw_max_pieces 10
                         --adj_draw_prob 0.9
                         --multi-pv-prob 0.35 
                         --multi-pv-eval-diff 55 
                         --multi-pv-min-pieces 10
                        ",
                        time, max_nodes, depth, worker_seed
                    )])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                    .expect("Failed to start process");
                let mut stdin = command.stdin.take().unwrap();
                let stdout = command.stdout.take().unwrap();
                let mut f = BufReader::new(stdout);

                //sending the child process id

                'outer: loop {
                    let mut start_pos = String::new();
                    {
                        if start_pos.is_empty() {
                            let guard = open.lock().unwrap();
                            let mut counter = op_counter.lock().unwrap();
                            if *counter >= guard.len() {
                                *counter = 0;
                            }

                            let opening = guard.get(*counter).unwrap();
                            start_pos = opening.clone();
                            *counter += 1;
                        }
                        if cfg!(debug_assertions) {
                            println!("Using the opening {start_pos}");
                        }
                    }
                    let trimmed = start_pos.trim().replace("\n", "");
                    stdin
                        .write_all((String::from(trimmed) + "\n").as_bytes())
                        .unwrap();
                    let mut game = Vec::new();
                    loop {
                        let mut buffer = String::new();
                        match f.read_line(&mut buffer) {
                            Ok(_) => {}
                            Err(e) => {
                                println!("{:?}", e)
                            }
                        }
                        buffer = buffer.trim().replace("\n", "");
                        if buffer != "BEGIN" && buffer != "END" {
                            game.push(String::from(buffer.trim().replace("\n", "")));
                        }
                        if buffer == "END" {
                            break;
                        }
                    }
                    let is_send = sender.send(game);
                    if let Err(_) = is_send {
                        break;
                    }
                    if counter.load(std::sync::atomic::Ordering::Relaxed) >= max_samples {
                        break;
                    }
                }
                stdin
                    .write_all((String::from("terminate\n")).as_bytes())
                    .unwrap();

                command.kill().unwrap();
            });
            handles.push(handle);
        }
        'game: for game in rx {
            let mut save_game = Game::new();
            for (index, value) in game.iter().rev().enumerate() {
                let splits: Vec<&str> = value.split("!").collect();
                let wp: u32 = splits[0].parse().expect("Could not parse white-pieces");
                let bp: u32 = splits[1].parse().expect("Could not parse black-pieces");
                let k: u32 = splits[2].parse().expect("Could not parse king-pieces");
                let color: i32 = splits[3].parse().expect("Could not parse color");
                let value: i32 = splits[5].parse().expect("Could not parse value");

                let mut position = Position::default();
                position.wp = wp;
                position.bp = bp;
                position.k = k;
                position.color = color as i8;

                /*position.print_position();
                println!("Value: {}",value);
                println!("Fenstring: {}",position.get_fen_string());
                println!("\n");
                */

                let result_string = String::from(splits[4].replace("\n", "").trim());
                if cfg!(debug_assertions) {
                    println!("{}", value);
                }
                //writing the samples to our file
                let mut sample = Sample::Sample::default();
                sample.position = position;
                sample.result = Sample::Result::from(result_string.as_str());

                if cfg!(debug_assertions) {
                    /*if sample.result == Sample::Result::UNKNOWN {
                        println!("Error {result_string}");
                    }
                    */
                }
                /*
                if sample.result == Sample::Result::UNKNOWN {
                    println!("Error UNKNOWN result");
                    println!("{:?}", game.first().unwrap());
                    continue 'game;
                }
                */
                total_count += 1;
                if index == 0 {
                    //setting the initial position and result for the game
                    save_game.set_start_position(sample.position, value as i16);
                    save_game.result = sample.result;
                    //println!("----------------------");
                    //sample.position.print_position();
                } else {
                    if save_game
                        .add_position(sample.position, value as i16)
                        .is_none()
                    {
                        println!("Could not add the position");
                        continue 'game;
                    }
                }
                let flipped_position = if sample.position.color == -1 {
                    sample.position.get_color_flip()
                } else {
                    sample.position
                };

                if !filter.check(&flipped_position) && !sample.position.has_capture() {
                    unique_count += 1;
                    bar.inc(1);
                    filter.set(&flipped_position);
                    thread_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if thread_counter.load(std::sync::atomic::Ordering::Relaxed) >= self.max_samples
                    {
                        break 'game;
                    }
                }
            }
            save_game
                .save_game(&mut writer)
                .expect("Could not save the game");
            average_game_length += save_game.moves.len() as f32;
            game_count += 1.0;
            let average = average_game_length / (game_count + 0.0001);
            bar.println(format!("average game length is {}", average));
        }

        for handle in handles {
            handle.join().unwrap();
        }
        println!(
            "We got back {} unique samples and a total of {} ",
            unique_count, total_count
        );
        writer.flush().expect("Could not flush writer");

        Ok(())
    }
}
