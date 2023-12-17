use crate::Pos;
use crate::Pos::Position;
use crate::Sample;
use bloomfilter::reexports::bit_vec::BitBlock;
use bloomfilter::Bloom;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use indicatif::{ProgressBar, ProgressStyle};
use mktemp::Temp;
use rand::seq::SliceRandom;
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::{BufRead, Seek, Write};
use std::option;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use Sample::SampleType;
//Rescorer takes fen/pdn strings and scores them
#[derive(Debug, Default)]
pub struct Rescorer {
    path: String,
    output: String,
    num_workers: usize,
    pub max_rescores: Option<usize>,
}

//Generator produces fen_strings
#[derive(Debug, Default)]
pub struct Generator {
    book: String,
    output: String,
    num_workers: usize,
    pub max_samples: usize,
}

pub fn merge_samples(samples: Vec<&str>, output: &str) -> std::io::Result<()> {
    let mut filter = Bloom::new_for_fp_rate(1000000000, 0.1);
    let mut writer = File::create(output)?;
    let mut unique_count: usize = 0;
    let mut total_count = 0;
    for file in samples {
        let mut reader = File::open(file)?;
        //first read the number of positions in the file
        let num_data = reader.read_u64::<LittleEndian>()?;

        for _ in 0..num_data {
            let mut sample = Sample::Sample::default();
            sample.read_into(&mut reader)?;
            sample.write_fen(&mut writer)?;
            if let Sample::SampleType::Fen(fen_string) = sample.position {
                if !filter.check(&fen_string) {
                    unique_count += 1;
                    filter.set(&fen_string);
                }
                total_count += 1;
            }
        }
    }
    drop(writer);
    let path = Path::new(output);
    prepend_file((total_count as u64).to_le_bytes().as_slice(), &path)?;
    println!("We processed {total_count} samples and found {unique_count} unique samples");
    Ok(())
}

fn prepend_file<P: AsRef<Path>>(data: &[u8], file_path: &P) -> std::io::Result<()> {
    let tmp_path = Temp::new_file()?;
    let mut tmp = File::create(&tmp_path)?;
    let mut src = File::open(&file_path)?;
    tmp.write_all(&data)?;
    std::io::copy(&mut src, &mut tmp)?;
    std::fs::remove_file(file_path)?;
    std::fs::rename(&tmp_path, file_path)?;

    Ok(())
}

pub fn create_unique_fens(in_str: &str, out: &str) -> std::io::Result<()> {
    //to be implemented
    let input = Path::new(in_str);
    let output = Path::new(out);
    let reader = BufReader::with_capacity(10000000, File::open(&input)?);
    let mut writer = File::create(&output)?;
    let mut filter = Bloom::new_for_fp_rate(1000000000, 0.1);
    let mut line_count: usize = 0;
    for line in reader.lines() {
        let fen_string = line?;
        let pos = Position::try_from(fen_string.as_str()).unwrap_or(Position::default());
        if pos == Position::default() {
            continue;
        }

        if !filter.check(&pos) {
            writer.write_all((fen_string + "\n").as_str().as_bytes())?;
            filter.set(&pos);
            line_count += 1;
        }
    }
    prepend_file(format!("{line_count}\n").as_str().as_bytes(), &output)?;
    Ok(())
}

pub fn count_positions<F: Fn(Position) -> bool>(
    path: String,
    predicate: F,
) -> std::io::Result<usize> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = String::new();
    reader.read_line(&mut buffer).unwrap();
    let bar = ProgressBar::new(buffer.replace("\n", "").parse::<u64>().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let mut counter: usize = 0;
    for line in reader.lines() {
        let pos = Position::try_from(line.unwrap().as_str())?;
        if predicate(pos) {
            counter += 1;
        }
        bar.inc(1);
    }

    Ok(counter)
}

pub fn count_material_less_than(path: String, count: usize) -> std::io::Result<usize> {
    count_positions(path, |pos| {
        (pos.bp.count_ones() + pos.wp.count_ones()) as usize <= count
    })
}

pub fn get_material_distrib(path: String) -> std::io::Result<HashMap<u32, usize>> {
    let mut my_map = HashMap::new();

    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = String::new();
    reader.read_line(&mut buffer).unwrap();
    let bar = ProgressBar::new(buffer.replace("\n", "").parse::<u64>().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    for line in reader.lines() {
        let pos = Position::try_from(line.unwrap().as_str())?;
        let piece_count = pos.bp.count_ones() + pos.wp.count_ones();
        *my_map.entry(piece_count).or_insert(0) += 1;
        bar.inc(1);
    }
    Ok(my_map)
}

//will be used to generate games/results
impl Generator {
    pub fn new(book: String, output: String, num_workers: usize, max_samples: usize) -> Generator {
        Generator {
            book,
            output,
            num_workers,
            max_samples,
        }
    }

    pub fn generate_games(self) -> std::io::Result<()> {
        //need a bloomfilter here
        let mut filter = Bloom::new_for_fp_rate(1000000000, 0.01);
        let thread_counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        let mut reader = BufReader::with_capacity(1000000, File::open(self.book.clone())?);
        let mut output = File::create(self.output.clone())?;
        let mut openings = Arc::new(Mutex::new(Vec::new()));
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

        for _ in 0..self.num_workers {
            let open = Arc::clone(&openings);
            let sender = tx.clone();
            let counter = Arc::clone(&thread_counter);
            let handle = std::thread::spawn(move || {
                let mut command = Command::new("./generator2")
                    .args(["--generate --time 10"])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                    .expect("Failed to start process");
                let mut stdin = command.stdin.take().unwrap();
                let stdout = command.stdout.take().unwrap();
                let mut f = BufReader::new(stdout);

                //get one random opening

                'outer: loop {
                    let mut start_pos = String::new();
                    {
                        while start_pos.is_empty() {
                            let guard = open.lock().unwrap();
                            let opening = guard.choose(&mut rand::thread_rng()).unwrap();
                            start_pos = opening.clone();
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
                    if counter.load(std::sync::atomic::Ordering::SeqCst) >= self.max_samples {
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
        let mut unique_count = 0;
        let mut total_count = 0;
        'game: for game in rx {
            for value in game {
                let splits: Vec<&str> = value.split("!").collect();
                let position = String::from(splits[0].replace("\n", "").trim());
                let result_string = String::from(splits[1].replace("\n", "").trim());
                if cfg!(debug_assertions) {
                    println!("{}", value);
                }

                if !filter.check(&position) {
                    unique_count += 1;
                    filter.set(&position);
                    bar.inc(1);

                    thread_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if thread_counter.load(std::sync::atomic::Ordering::SeqCst) >= self.max_samples
                    {
                        break 'game;
                    }
                }
                //writing the samples to our file
                let mut sample = Sample::Sample::default();
                sample.position = SampleType::Fen(position);
                sample.result = Sample::Result::from(result_string.as_str());
                if sample.result == Sample::Result::UNKNOWN {
                    println!("Error {result_string}");
                }
                if sample.result != Sample::Result::UNKNOWN {
                    sample.write_fen::<File>(&mut output)?;
                    total_count += 1;
                }
            }
        }

        for handle in handles {
            handle.join().unwrap();
        }
        println!("We got back {unique_count} unique samples and a total of {total_count} ");
        //writing the file
        drop(output);
        let path = Path::new(self.output.as_str());
        prepend_file((total_count as u64).to_le_bytes().as_slice(), &path)?;
        Ok(())
    }
}
