use crate::Move;
use crate::Pos::Position;
use crate::Pos::Square;
use crate::Sample;
use crate::Sample::Game;
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
use rayon::prelude::*;
use rip_shuffle::RipShuffleParallel;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
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
//Generator produces fen_strings
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
            let mut command = Command::new("./generator2")
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

pub fn merge_rescored_data(input: Vec<&str>, output: &str) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(output)?);
    let mut filter = Bloom::new_for_fp_rate(1000000000, 0.01);
    let mut total_count = 0;
    let mut unique_count = 0;
    for path in input.iter() {
        let mut reader = BufReader::new(File::open(path)?);
        for sample in reader.iter_samples() {
            match sample.result {
                Result::TBDRAW | Result::TBLOSS | Result::TBWIN => {
                    if !filter.check(&sample.position) {
                        filter.set(&sample.position);
                        sample.write_fen(&mut writer)?;
                        unique_count += 1;
                        total_count += 1;
                    }
                }
                Result::UNKNOWN => {}
                _ => {
                    if !filter.check(&sample.position) {
                        filter.set(&sample.position);
                        unique_count += 1;
                    }

                    total_count += 1;
                    sample.write_fen(&mut writer)?;
                }
            }
        }
    }

    writer.flush()?;

    println!(
        "Got back {} unique samples while processing {} samples",
        unique_count, total_count
    );

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

pub fn create_subset(input: &str, output: &str, num_samples: usize) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(output)?);
    let mut reader = BufReader::new(File::open(input)?);
    for sample in reader.iter_samples().take(num_samples) {
        sample.write_fen(&mut writer)?;
    }
    Ok(())
}

//Refactoring this as well
pub fn create_unique_fens(in_str: &str, out: &str) -> std::io::Result<()> {
    //to be implemented
    let input = Path::new(in_str);
    let output = Path::new(out);
    let reader = BufReader::with_capacity(10000000, File::open(&input)?);
    let mut writer = BufWriter::new(File::create(&output)?);
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
    Ok(())
}

pub fn count_unique_samples(input: &str) -> std::io::Result<usize> {
    let mut reader = BufReader::new(File::open(input)?);
    let filter: RefCell<Bloom<Sample::Sample>> =
        RefCell::new(Bloom::new_for_fp_rate(1000000000, 0.01));
    Ok(reader
        .iter_samples()
        .filter(|sample| !filter.borrow().check(&sample))
        .map(|sample| filter.borrow_mut().set(&sample))
        .count())
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

pub fn print_samples(path: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let game_iter = reader.iter_games();

    for game in game_iter {
        let samples = game.get_samples();
        for sample in samples.iter() {
            println!("-------------------");
            sample.position.print_position();
            println!("Mover: {} Value: {}", sample.position.color, sample.value);
        }
    }
    Ok(())
}
//#[cfg(target_os = "windows")]
/*pub fn create_mlh_data(path: &str, output: &str, base: &TableBase::Base) -> std::io::Result<()> {
    let mut reader = BufReader::with_capacity(1000000, File::open(path)?);
    let mut writer = BufWriter::with_capacity(10000, File::create(output)?);

    let mut command = Command::new("./generator2")
        .args(["--eval-loop"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to start process");
    let mut stdin = command.stdin.take().unwrap();
    let stdout = command.stdout.take().unwrap();
    let mut f = BufReader::new(stdout);

    for game in reader.iter_games() {
        let mut mlh_counter = 0;
        for sample in game.iter() {
            if !sample.position.has_capture() {
                let probe = base.probe_with_position(sample.position).unwrap();
                let mut eval: i32 = 0;

                if probe == Result::TBDRAW {
                    stdin
                        .write_all((sample.position.clone().get_fen_string() + "\n").as_bytes())
                        .unwrap();

                    let mut buffer = String::new();
                    match f.read_line(&mut buffer) {
                        Ok(_) => {}
                        Err(e) => {
                            println!("{:?}", e)
                        }
                    }
                    buffer = buffer.trim().replace("\n", "");

                    eval = buffer.parse().unwrap_or(0);
                } else if probe == Result::TBWIN || probe == Result::TBLOSS {
                    eval = 1000;
                    //trying out dtw once again
                }

                if eval.abs() >= 500 {
                    let mut copy = sample.clone();
                    if copy.position.color == -1 {
                        copy.position = copy.position.get_color_flip();
                    }
                    copy.mlh = mlh_counter as i16;
                    copy.write_fen(&mut writer)
                        .expect("Error writing sample to a file");
                }
            }
            mlh_counter += 1;
        }
    }
    writer.flush()?;
    Ok(())
}
*/

/*pub fn convert_samples_to_games(path: &str, output: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = BufWriter::new(File::create(output)?);

    'outer: for game in reader.iter_games() {
        //old format - samples are in reverse order
        let mut new_game = Game::new();
        for (index, sample) in game.iter().rev().enumerate() {
            if index == 0 {
                new_game.set_start_position(sample.position);
                new_game.set_result(sample.result);
            } else {
                let added = new_game.add_position(sample.position);
                if added == None {
                    continue 'outer;
                }
            }
        }
        //now we can write the game to a file
        new_game.save_game(&mut writer)?;
    }
    Ok(())
}
*/

pub fn filter_training_data(path: &str, out: &str) -> std::io::Result<()> {
    let thresh_hold = 0.1;
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = BufWriter::new(File::create(out)?);
    let mut counter = 0;
    let mut sample_iter = reader.iter_samples();
    let mut rng = thread_rng();
    let distrib: Uniform<f64> = Uniform::new(0.0, 1.0);
    for s in sample_iter {
        let mut sample = s;
        if sample.result == Result::TBWIN {
            sample.value = 10000;
        } else if sample.result == Result::TBLOSS {
            sample.value = -10000;
        } else if sample.result == Result::TBDRAW {
            sample.value = 0;
        }
        sample.write_fen(&mut writer)?;
    }
    println!("{}", counter);
    Ok(())
}

//#[cfg(target_os = "windows")]
pub fn rescore_games(path: &str, output: &str, base: &TableBase::Base) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = BufWriter::new(File::create(output)?);
    let mut filter = Bloom::new_for_fp_rate(1000000000, 0.01);
    let mut total_count = 0;
    let mut written_count: u64 = 0;
    for game in reader.iter_games() {
        if game.result == Result::UNKNOWN {
            continue;
        }
        let samples = rescore_game(&game, base)?;

        for sample in samples {
            if sample.position.has_capture() {
                continue;
            }
            if (sample.position.bp == 0) || (sample.position.wp == 0) {
                continue;
            }
            total_count += 1;
            match sample.result {
                Result::TBDRAW | Result::TBLOSS | Result::TBWIN => {
                    if !filter.check(&sample.position) {
                        filter.set(&sample.position);
                        sample.write_fen(&mut writer)?;
                        written_count += 1;
                    }
                }
                Result::UNKNOWN => {}
                _ => {
                    sample.write_fen(&mut writer)?;
                    written_count += 1;
                }
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

pub fn rescore_game(game: &Game, base: &TableBase::Base) -> std::io::Result<Vec<Sample::Sample>> {
    let mut game_samples = game.get_samples();
    let mut rng = thread_rng();
    let mut adj_prob = 0.9;
    let mut uniform = Uniform::new(0.0, 1.0);
    let last_position = game_samples.last().unwrap().position;
    if last_position.bp == 0 || last_position.wp == 0 {
        game_samples.pop().expect("Game was empty");
    }

    let last = game_samples.last().expect("Game was empty");
    let mut local_result = last.result;
    for sample in game_samples.iter_mut().rev() {
        let mover = sample.position.color;
        if mover == -1 {
            sample.position = sample.position.get_color_flip();
        }

        let result = base.probe_with_position(sample.position).unwrap();
        if result == Result::UNKNOWN {
            sample.result = local_result;
        } else {
            sample.result = result;
            local_result = match result {
                Result::TBWIN => Result::WIN,
                Result::TBLOSS => Result::LOSS,
                Result::TBDRAW => Result::DRAW,
                _ => local_result,
            };
        }
        local_result = match local_result {
            Result::WIN | Result::TBWIN => Result::LOSS,
            Result::LOSS | Result::TBLOSS => Result::WIN,
            _ => local_result,
        };
    }
    //doing some sort of draw-adjudication
    let mut filter_samples = Vec::new();
    let mut count = 0;
    let mut sum_last = 0;
    const adj_moves: i32 = 10;
    for s in game_samples.iter().cloned() {
        let mut sample = s;
        let piece_count = sample.position.piece_count();
        if piece_count <= 10 {
            sum_last += sample.value.abs() as i32;
            count += 1;
        }

        if count >= adj_moves {
            let avg = sum_last.div(count);
            if avg.abs() <= 1 && sample.result == Result::DRAW {
                let uni_num = uniform.sample(&mut rng);
                if uni_num <= adj_prob {
                    break;
                }
            }
            count = 0;
            sum_last = 0;
        }
        if sample.result == Result::TBWIN {
            sample.value = 10000;
        } else if sample.result == Result::TBLOSS {
            sample.value = -10000;
        } else if sample.result == Result::TBDRAW {
            sample.value = 0;
        }

        filter_samples.push(sample);
    }
    Ok(filter_samples)
}

/*pub fn create_policy_data(path: &str, output: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = BufWriter::new(File::create(output)?);
    for game in reader.iter_games() {
        //need to use the chinook format
        //or keep using fen-strings
        for window in game.windows(2) {
            let next_pos = window[0].position;
            let prev_pos = window[1].position;
            if prev_pos.has_capture() {
                continue;
            }
            if (prev_pos.bp == 0) || (prev_pos.wp == 0) {
                continue;
            }

            let move_encoding;
            if prev_pos.color == -1 {
                move_encoding = Move::get_move_encoding_from_pos(
                    prev_pos.get_color_flip(),
                    next_pos.get_color_flip(),
                )
                .unwrap_or(-1);
            } else {
                move_encoding = Move::get_move_encoding_from_pos(prev_pos, next_pos).unwrap_or(-1);
            }

            if move_encoding >= 0 {
                let mut sample = window[1].clone();
                if sample.position.color == -1 {
                    sample.position = sample.position.get_color_flip();
                }

                sample.mlh = move_encoding as i16;
                sample.write_fen(&mut writer)?;
            }
        }
    }

    Ok(())
}
*/

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

    //storing bloomfilters instead of scanning the previous file -> need to store total_count and
    //unique_count as well

    fn store_bloom<S, T>(filter: Bloom<S>, Bloomoutput: &T) -> std::io::Result<()>
    where
        T: Write,
        S: Hash,
    {
        //check how to get the state of the bloom-filter
        Ok(())
    }

    pub fn generate_games(&self) -> std::io::Result<()> {
        let mut filter = Bloom::new_for_fp_rate(3000000000, 0.01);
        let mut unique_count = 0;
        let mut total_count = 0;
        let time = self.time;
        let max_nodes = self.max_nodes;
        let depth = self.depth;
        let mut writer = BufWriter::new(File::create(self.output.clone())?);
        let thread_counter = Arc::new(AtomicUsize::new(0));
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
        let max_samples = self.max_samples;
        for _id in 0..self.num_workers {
            let open = Arc::clone(&openings);
            let sender = tx.clone();
            let counter = Arc::clone(&thread_counter);
            let handle = std::thread::spawn(move || {
                let mut command = Command::new("./generator2")
                    .args([format!(
                        "--generate --time {} --nodes {} --depth {}",
                        time, max_nodes, depth
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

                let result_string = String::from(splits[4].replace("\n", "").trim());
                if cfg!(debug_assertions) {
                    println!("{}", value);
                }
                //writing the samples to our file
                let mut sample = Sample::Sample::default();
                sample.position = position;
                sample.result = Sample::Result::from(result_string.as_str());

                if cfg!(debug_assertions) {
                    if sample.result == Sample::Result::UNKNOWN {
                        println!("Error {result_string}");
                    }
                }
                if sample.result == Sample::Result::UNKNOWN {
                    println!("Error UNKNOWN result");
                    println!("{:?}", game.first().unwrap());
                    continue 'game;
                }
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
                if !filter.check(&sample.position) && !sample.position.has_capture() {
                    unique_count += 1;
                    bar.inc(1);
                    filter.set(&sample.position);
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
