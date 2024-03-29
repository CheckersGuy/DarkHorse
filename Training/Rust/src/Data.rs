use crate::Pos::Position;
use crate::Sample;
use crate::Sample::SampleIteratorTrait;
use crate::TableBase;
use bloomfilter::reexports::bit_vec::BitBlock;
use bloomfilter::Bloom;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use rip_shuffle::RipShuffleParallel;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::hash::Hash;
use std::io::{BufRead, Write};
use std::io::{BufReader, BufWriter};
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

pub fn material_distrib(path: &str) -> std::io::Result<HashMap<usize, usize>> {
    let mut filter = Bloom::new_for_fp_rate(3000000000, 0.01);
    let mut my_map = HashMap::new();
    let mut reader = BufReader::new(File::open(path)?);
    let _buffer = String::new();

    for sample in reader.iter_samples() {
        let pos = match sample.position.clone() {
            SampleType::Fen(fen_string) => (fen_string, sample.position.get_squares()),
            SampleType::Pos(_) => continue,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "didnt find a position in sample",
                ))
            }
        };
        if !filter.check(&pos.0) {
            filter.set(&pos.0);
            let piece_count = pos.1.unwrap().len();
            *my_map.entry(piece_count).or_insert(0) += 1;
        }
    }
    Ok(my_map)
}
pub fn dump_mlh_samples(input: &str, output: &str) -> std::io::Result<()> {
    let mut filter = Bloom::new_for_fp_rate(1000000000, 0.01);
    let mut writer = BufWriter::new(File::create(output)?);
    let mut total_counter: u64 = 0;
    let mut reader = BufReader::new(File::open(input)?);

    for sample in reader.iter_samples() {
        let fen_string = match sample.position.clone() {
            Sample::SampleType::Fen(fen_string) => fen_string,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Wrong data format",
                ));
            }
        };

        if !filter.check(&fen_string) {
            if sample.mlh > 0 {
                sample.write_fen(&mut writer)?;
                total_counter += 1;
            }
            filter.set(&fen_string);
        }
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
//needs to be reworked
//#[cfg(target_os = "windows")]
pub fn create_mlh_data(path: &str, output: &str, base: &TableBase::Base) -> std::io::Result<()> {
    let mut reader = BufReader::with_capacity(1000000, File::open(path)?);
    let mut writer = BufWriter::with_capacity(100000, File::create(output)?);
    let mut write_count = 0;
    for game in reader.iter_games() {
        let mut mlh_counter = None;
        for sample in game.iter() {
            let squares = sample.position.get_squares().unwrap();
            if squares.len() > 10 {
                continue;
            }
            let fen_string = match sample.position {
                Sample::SampleType::Fen(ref fen) => fen,
                _ => return Ok(()),
            };
            let probe = base.probe_dtw(fen_string);
            if let Ok(Some(count)) = probe {
                mlh_counter = Some(count);
            } else {
                if let Some(count) = mlh_counter {
                    match sample.result {
                        Sample::Result::TBWIN
                        | Sample::Result::TBLOSS
                        | Sample::Result::WIN
                        | Sample::Result::LOSS => {
                            mlh_counter = Some(count + 1);
                        }
                        _ => mlh_counter = None,
                    }
                }
            }
            if let Some(count) = mlh_counter {
                let mut copy = sample.clone();
                copy.mlh = count as i16;
                write_count += 1;
                copy.write_fen(&mut writer)?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

//#[cfg(target_os = "windows")]
pub fn rescore_game(game: &mut Vec<Sample::Sample>, base: &TableBase::Base) {
    let get_mover = |fen: &str| -> i32 {
        match fen.chars().next() {
            Some('W') => 1,
            Some('B') => -1,
            _ => 0,
        }
    };
    let mut counter = 0;
    let mut mlh_counter: Option<i32> = None;
    for sample in game.iter_mut() {
        let fen_string = match sample.position {
            Sample::SampleType::Fen(ref fen) => fen,
            _ => return,
        };
        let probe = base.probe_dtw(fen_string);
        if let Ok(Some(count)) = probe {
            mlh_counter = Some(count);
        } else {
            if let Some(count) = mlh_counter {
                mlh_counter = Some(count + 1);
            }
        }
        if get_mover(fen_string) == 1 {
            counter += 1;
        }
        if let Some(count) = mlh_counter {
            sample.mlh = count as i16;
        } else {
            sample.mlh = -1000;
        }
    }
    if counter == game.len() {
        //game has previously been rescored
        return;
    }

    let last = game.last().unwrap().clone();
    let fen_string = match last.position {
        Sample::SampleType::Fen(ref fen) => fen,
        _ => return,
    };

    let mut local_result = (get_mover(fen_string), last.result);
    for sample in game.iter_mut() {
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
    let mut writer = BufWriter::new(File::create(output)?);
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
    writer.flush();

    println!(
        "Got back a total of {} while processing {} samples",
        written_count, total_count
    );
    Ok(())
}

pub fn create_policy_data(path: &str, output: &str, base: &TableBase::Base) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut total_count = 0;
    let mut writer = BufWriter::new(File::create(output)?);
    for game in reader.iter_games() {
        for window in game.windows(2) {
            let fen_next = match window[0].position {
                SampleType::Fen(ref fen_string) => fen_string.clone(),
                _ => String::new(),
            };
            let fen_previous = match window[1].position {
                SampleType::Fen(ref fen_string) => fen_string.clone(),
                _ => String::new(),
            };
            let move_encoding = base
                .get_move_encoding(fen_previous.as_str(), fen_next.as_str())
                .unwrap();

            if move_encoding > 0 {
                //base.print_fen(fen_previous.as_str()).unwrap();
                //println!("{move_encoding}");
                let mut sample = window[1].clone();
                sample.mlh = move_encoding as i16;
                sample.write_fen(&mut writer)?;
                total_count += 1;
            }
        }
    }

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

    fn load_previous_file(&self) -> std::io::Result<(u64, u64, Bloom<String>)> {
        let mut filter = Bloom::new_for_fp_rate(3000000000, 0.01);
        let mut unique_count = 0;
        let mut total_count = 0;
        let mut writer = BufWriter::new(File::create(self.output.clone()).unwrap());
        if self.prev_file == None {
            return Ok((unique_count, total_count, filter));
        }
        //we iterate over all samples and build up the filters from there
        let mut reader = BufReader::new(File::open(self.prev_file.unwrap())?);
        let iterator = reader.iter_samples();

        for sample in iterator {
            let pos_string = match sample.position {
                Sample::SampleType::Fen(ref fen_string) => fen_string,
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "No previous file",
                    ))
                }
            };

            match sample.result {
                Result::TBWIN | Result::TBLOSS | Result::TBDRAW => {
                    if !filter.check(&pos_string) {
                        sample.write_fen(&mut writer)?;
                    }
                }
                _ => {
                    sample.write_fen(&mut writer)?;
                }
            }
            if !filter.check(&pos_string) {
                unique_count += 1;
                filter.set(&pos_string);
            }
            total_count += 1;
        }
        writer.flush().expect("Flush Error");
        println!(
            "Read a previous file with {} unique samples and {} total samples",
            unique_count, total_count
        );
        //checking and testing this stuff
        Ok((unique_count, total_count, filter))
    }

    pub fn generate_games(&self) -> std::io::Result<()> {
        let (mut unique_count, mut total_count, mut filter) = self.load_previous_file()?;
        let output_file = self.output.clone();
        let time = self.time;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .append(true)
                .open(self.output.clone())
                .unwrap(),
        );

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
            for value in game {
                let splits: Vec<&str> = value.split("!").collect();
                let position = String::from(splits[0].replace("\n", "").trim());
                let result_string = String::from(splits[1].replace("\n", "").trim());
                if cfg!(debug_assertions) {
                    println!("{}", value);
                }
                let pos = Position::try_from(position.as_str())?;
                let has_captures: bool = pos.get_jumpers::<1>() != 0;

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
                        if !filter.check(&position) {
                            sample.write_fen::<BufWriter<File>>(&mut writer)?;
                        }
                    } else {
                        sample.write_fen::<BufWriter<File>>(&mut writer)?;
                    }
                    total_count += 1;
                    if !filter.check(&position) {
                        unique_count += 1;
                        filter.set(&position);
                        bar.inc(1);
                        thread_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if thread_counter.load(std::sync::atomic::Ordering::Relaxed)
                            >= self.max_samples
                        {
                            break 'game;
                        }
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
        writer.flush().expect("Could not flush writer");

        Ok(())
    }
}
