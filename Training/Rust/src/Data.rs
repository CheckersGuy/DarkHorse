use crate::Pos::Position;
use crate::Sample;
use crate::Sample::SampleIteratorTrait;
use crate::TableBase;
use bloomfilter::reexports::bit_vec::BitBlock;
use bloomfilter::Bloom;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use indicatif::{ProgressBar, ProgressStyle};
use mktemp::Temp;
use rand::seq::SliceRandom;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::{BufRead, Write};
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
    drop(writer);
    prepend_file(format!("{u_count}\n").as_str().as_bytes(), &output)?;
    Ok(())
}

pub fn material_distrib(path: &str) -> std::io::Result<HashMap<u32, usize>> {
    let mut my_map = HashMap::new();
    let mut reader = BufReader::new(File::open(path)?);
    let _buffer = String::new();
    let num_samples = reader.read_u64::<LittleEndian>()?;
    let bar = ProgressBar::new(num_samples);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    for _ in 0..num_samples {
        let mut sample = Sample::Sample::default();
        sample.read_into(&mut reader)?;
        let pos: Position;

        match sample.position {
            SampleType::Fen(fen_string) => pos = Position::try_from(fen_string.as_str())?,
            SampleType::Pos(_) => continue,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "didnt find a position in sample",
                ))
            }
        }

        let piece_count = pos.bp.count_ones() + pos.wp.count_ones();
        *my_map.entry(piece_count).or_insert(0) += 1;
        bar.inc(1);
    }
    Ok(my_map)
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
//Refactoring this as well
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

#[cfg(target_os = "windows")]
pub fn rescore_game(game: &mut Vec<Sample::Sample>, base: &TableBase::Base) {
    let get_mover = |fen: &str| -> i32 {
        match fen.chars().next() {
            Some('W') => 1,
            Some('B') => -1,
            _ => 0,
        }
    };

    let last = game.last().unwrap().clone();
    let fen_string = match last.position {
        Sample::SampleType::Fen(ref fen) => fen,
        _ => return,
    };
    let mut local_result = (get_mover(fen_string), last.result);

    for sample in game {
        //probing tablebase
        let fen_string = match sample.position {
            Sample::SampleType::Fen(ref fen) => fen,
            _ => return,
        };
        let mover = get_mover(fen_string);
        let result = (mover, base.probe(fen_string).unwrap());
        if result.1 != Result::UNKNOWN {
            local_result = result;
        }

        let piece_count = sample.position.get_squares().unwrap().len();
        let mut adj_result;
        if piece_count > 10 {
            adj_result = match local_result.1 {
                Result::TBWIN => Result::WIN,
                Result::TBLOSS => Result::LOSS,
                Result::TBDRAW => Result::DRAW,
                _ => local_result.1,
            }
        } else {
            adj_result = local_result.1;
        }
        if mover != local_result.0 {
            adj_result = !adj_result;
        }
        if mover == -1 {
            sample.position = SampleType::Fen(SampleType::invert_fen_string(fen_string).unwrap());
        }
        sample.result = adj_result;
    }
}
#[cfg(target_os = "windows")]
pub fn rescore_games(path: &str, output: &str, base: &TableBase::Base) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = File::create(output)?;
    let mut filter = Bloom::new_for_fp_rate(1000000000, 0.01);
    let mut total_count = 0;
    let mut written_count: u64 = 0;
    for game in reader.iter_games() {
        let mut borrow_game = game.clone();
        rescore_game(&mut borrow_game, base);
        for sample in borrow_game {
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
    drop(writer);
    let path = Path::new(output);
    prepend_file((written_count as u64).to_le_bytes().as_slice(), &path)?;
    println!(
        "Got back a total of {} while processing {} samples",
        written_count, total_count
    );
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
            prev_file: None,
        }
    }

    fn load_previous_file(&self) -> std::io::Result<(u64, u64, Bloom<String>, Bloom<String>)> {
        let mut end_filter = Bloom::new_for_fp_rate(1000000000, 0.01);
        let mut filter = Bloom::new_for_fp_rate(1000000000, 0.01);

        let mut unique_count = 0;
        let mut total_count = 0;
        let mut writer = File::create(self.output.clone()).unwrap();
        if self.prev_file == None {
            return Ok((unique_count, total_count, filter, end_filter));
        }

        //we iterate over all samples and build up the filters from there

        let mut reader = BufReader::new(File::open(self.prev_file.unwrap())?);

        for sample in reader.iter_samples() {
            let pos_string = match sample.position {
                Sample::SampleType::Fen(ref fen_string) => fen_string,
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "No previous file",
                    ))
                }
            };
            if !filter.check(&pos_string) {
                unique_count += 1;
                filter.set(&pos_string);
            }
            match sample.result {
                Result::TBWIN | Result::TBLOSS | Result::TBDRAW => {
                    if !end_filter.check(&pos_string) {
                        end_filter.set(&pos_string);
                        sample.write_fen(&mut writer)?;
                    }
                }
                _ => {
                    sample.write_fen(&mut writer)?;
                }
            }
            total_count += 1;
        }

        println!(
            "Read a previous file with {} unique samples and {} total samples",
            unique_count, total_count
        );
        //checking and testing this stuff
        Ok((unique_count, total_count, filter, end_filter))
    }

    pub fn generate_games(&self) -> std::io::Result<()> {
        //need a bloomfilter here
        //load a previous file if present
        let (mut unique_count, mut total_count, mut filter, mut end_filter) =
            self.load_previous_file()?;
        let output_file = self.output.clone();
        let time = self.time;
        let mut writer = OpenOptions::new()
            .write(true)
            .append(true)
            .open(self.output.clone())
            .unwrap();

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
                    .args([format!("--generate --time {}", time)])
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
                    if counter.load(std::sync::atomic::Ordering::SeqCst) >= max_samples {
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
            for value in game {
                let splits: Vec<&str> = value.split("!").collect();
                let position = String::from(splits[0].replace("\n", "").trim());
                let result_string = String::from(splits[1].replace("\n", "").trim());
                if cfg!(debug_assertions) {
                    println!("{}", value);
                }
                let pos = Position::try_from(position.as_str())?;
                let has_captures: bool = pos.get_jumpers::<1>() != 0;
                if !filter.check(&position) && !has_captures {
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
                sample.position = SampleType::Fen(position.clone());
                sample.result = Sample::Result::from(result_string.as_str());

                if cfg!(debug_assertions) {
                    if sample.result == Sample::Result::UNKNOWN {
                        println!("Error {result_string}");
                    }
                }
                if sample.result != Sample::Result::UNKNOWN && !has_captures {
                    if result_string.starts_with("TB_") {
                        if !end_filter.check(&position) {
                            sample.write_fen::<File>(&mut writer)?;
                            end_filter.set(&position);
                            total_count += 1;
                        }
                    } else {
                        sample.write_fen::<File>(&mut writer)?;
                        total_count += 1;
                    }
                }
            }
        }

        for handle in handles {
            handle.join().unwrap();
        }
        println!(
            "We got back {} unique samples and a total of {} ",
            unique_count, total_count
        );
        drop(writer);
        let path = Path::new(output_file.as_str());
        prepend_file((total_count as u64).to_le_bytes().as_slice(), &path)?;
        Ok(())
    }
}
