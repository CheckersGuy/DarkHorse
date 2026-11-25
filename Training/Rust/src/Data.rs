#![feature(iter_array_chunks)]
use crate::Move;
use crate::Pos::MoveList;
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
use hyperloglog_rs::prelude::*;
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
use std::collections::TryReserveError;
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
use std::time::{Duration, Instant};
use Sample::{Result, SampleType};

use byteorder::WriteBytesExt;
use std::io::{Seek, SeekFrom};
use Sample::{parse_children_line, write_policy_header, PolicySample};

fn read_policy_partition(path: &str) -> std::io::Result<Vec<PolicySample>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    loop {
        let mut ps = PolicySample::new(Sample::Sample::default());
        match ps.read_into(&mut r) {
            Ok(()) => out.push(ps),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// Deletes the temporary partition files `<prefix>0 .. <prefix>{partitions-1}`.
/// Only call this after the merged output has been written and flushed.
fn remove_partitions(prefix: &str, partitions: usize) -> std::io::Result<()> {
    for i in 0..partitions {
        std::fs::remove_file(format!("{}{}", prefix, i))?;
    }
    Ok(())
}

pub fn rescoring_children(
    paths: Vec<&str>,
    output: &str,
    time: i32,
    child_time: i32,
    num_workers: usize,
    partitions: usize,
    queue_capacity: usize,
) -> std::io::Result<()> {
    let budget = format!("--time {} --child-time {}", time, child_time);
    rescoring_pol1(paths, output, &budget, num_workers, partitions, queue_capacity, true)
}

/// Default node budget per phase (pieces >= 17, 13-16, 9-12, <= 8) for the
/// node-based rescoring (engine --nodes_phase), see Training/value_net_plan.md.
pub const DEFAULT_NODES_PHASE: &str = "2500,2500,1500,750";

/// rescoring_children with a node budget per position instead of a time limit
/// (reproducible labels, independent of the CPU load).
pub fn rescoring_children_nodes(
    paths: Vec<&str>,
    output: &str,
    nodes_phase: &str,
    num_workers: usize,
    partitions: usize,
    queue_capacity: usize,
) -> std::io::Result<()> {
    let budget = format!("--nodes_phase {}", nodes_phase);
    rescoring_pol1(paths, output, &budget, num_workers, partitions, queue_capacity, true)
}

/// Value-net rescoring: root value only (no child searches). As in rescoring_children
/// the root is searched without MLH (engine --no_mlh), so the value label is the pure
/// search value (the engine adds the moves-left bonus itself); child scores keep MLH.
/// Output: POL1 file with empty child lists (read by the value loader, fill_value).
pub fn rescoring_value(
    paths: Vec<&str>,
    output: &str,
    nodes_phase: &str,
    num_workers: usize,
    partitions: usize,
    queue_capacity: usize,
) -> std::io::Result<()> {
    let budget = format!("--nodes_phase {}", nodes_phase);
    rescoring_pol1(paths, output, &budget, num_workers, partitions, queue_capacity, false)
}

fn rescoring_pol1(
    paths: Vec<&str>,
    output: &str,
    budget: &str, // engine options: "--time T --child-time C" or "--nodes_phase A,B,C,D"
    num_workers: usize,
    partitions: usize,
    queue_capacity: usize,
    children: bool,
) -> std::io::Result<()> {
    const AVG_CHILDREN: f64 = 8.0;
    let to_rescore = count_positions_to_rescore(&paths, RESCORE_VALUE_THRESHOLD)?;
    let paths = expand_game_paths(&paths)?;

    // rough estimate: ~1 ms per search (time mode, or ~1500 nodes)
    let per_pos_ms = 1.0 + if children { AVG_CHILDREN } else { 0.0 };
    let est_hours = to_rescore as f64 * per_pos_ms / 1000.0 / num_workers as f64 / 3600.0;
    println!("{} positions, estimated {:.2} hours ({})", to_rescore, est_hours, budget);

    let (work_tx, work_rx) = mpsc::sync_channel::<PolicySample>(queue_capacity);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (result_tx, result_rx) = mpsc::sync_channel::<PolicySample>(queue_capacity);

    let bar = ProgressBar::new(to_rescore);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise}] {bar:40.cyan/blue} {pos:>9}/{len:9} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    // --- writer thread: random partition files, no header ---
    let writer_handle = {
        let mut files: Vec<BufWriter<File>> = Vec::new();
        for i in 0..partitions {
            files.push(BufWriter::new(File::create(format!("{}{}", output, i))?));
        }
        thread::spawn(move || {
            let mut rng = StdRng::from_rng(thread_rng()).unwrap();
            for ps in result_rx.iter() {
                let p = rng.gen::<usize>() % partitions;
                ps.write(&mut files[p]).expect("Could not write sample");
            }
            for f in files.iter_mut() {
                f.flush().expect("flush");
            }
        })
    };

    // --- workers ---
    let mut worker_handles = Vec::new();
    for worker_id in 0..num_workers {
        let work_rx = Arc::clone(&work_rx);
        let result_tx = result_tx.clone();
        let worker_bar = bar.clone();
        let budget = budget.to_string();
        worker_handles.push(thread::spawn(move || {
            let engine_args = if children {
                format!("--eval-loop --children --no_mlh {} --hash_size 64", budget)
            } else {
                format!("--eval-loop --no_mlh {} --hash_size 64", budget)
            };
            let mut command = Command::new("./MainEngine")
                .args(engine_args.split_whitespace())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("Failed to start process");
            let mut stdin = command.stdin.take().unwrap();
            let mut f = BufReader::new(command.stdout.take().unwrap());
            // root values are searched without MLH (both modes; child scores keep MLH):
            // the engine must confirm --no_mlh with a first line "no_mlh" (an old
            // engine ignores the option and would rescore the root with MLH)
            let mut expect_banner = true;

            loop {
                let received = { work_rx.lock().unwrap().recv() };
                let mut ps = match received {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let fen = ps.sample.position.get_fen_string();
                if stdin.write_all((fen + "\n").as_bytes()).is_err() {
                    break;
                }

                let mut buf = String::new();
                if expect_banner {
                    if f.read_line(&mut buf).is_err() || buf.trim() != "no_mlh" {
                        panic!(
                            "./MainEngine does not support --no_mlh (first line '{}'): \
                             update the engine copy in Training/Rust",
                            buf.trim()
                        );
                    }
                    expect_banner = false;
                    buf.clear();
                }
                if f.read_line(&mut buf).is_err() {
                    continue;
                }
                if !children {
                    match buf.trim().parse::<i16>() {
                        Ok(v) => ps.sample.value = v,
                        Err(_) => {
                            println!("Worker {worker_id} could not parse '{}'", buf.trim());
                            continue;
                        }
                    }
                } else {
                    match parse_children_line(&buf) {
                        Some((root, children)) => {
                            ps.sample.value = root;
                            ps.children = children;
                        }
                        None => {
                            println!("Worker {worker_id} could not parse '{}'", buf.trim());
                            continue;
                        }
                    }
                }
                worker_bar.inc(1);
                if result_tx.send(ps).is_err() {
                    break;
                }
            }
            let _ = stdin.write_all(b"terminate\n");
            let _ = command.wait();
        }));
    }

    // --- producer: same filters as rescoring_data ---
    let mut filter = Bloom::new_for_fp_rate(4_000_000_000, 0.01);
    for path in paths.iter() {
        let mut reader = BufReader::new(File::open(path)?);
        for sample in reader.iter_samples() {
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

            let ps = PolicySample::new(sample);
            if sample.value.abs() >= RESCORE_VALUE_THRESHOLD {
                // keep the old value, no children
                if result_tx.send(ps).is_err() {
                    break;
                }
            } else if work_tx.send(ps).is_err() {
                break;
            }
        }
    }
    drop(work_tx);
    drop(result_tx);
    for h in worker_handles {
        h.join().unwrap();
    }
    writer_handle.join().unwrap();
    bar.finish();

    // --- shuffle partitions into the final file (with header) ---
    let mut writer = BufWriter::new(File::create(output)?);
    write_policy_header(&mut writer, 0)?;
    let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    let mut count: u64 = 0;
    for i in 0..partitions {
        let mut samples = read_policy_partition(&format!("{}{}", output, i))?;
        samples.par_shuffle(&mut rng);
        for ps in samples {
            ps.write(&mut writer)?;
            count += 1;
        }
    }
    writer.flush()?;
    writer.seek(SeekFrom::Start(4))?; // patch the count after the magic
    writer.write_u64::<LittleEndian>(count)?;
    writer.flush()?;
    remove_partitions(output, partitions)?;
    Ok(())
}

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
    /// engine sampling (see Reservoir in GameLogic.h): keep probability per new
    /// search-tree position, cap per game, optional keep factors per |eval| band
    pub sample_prob: f64,
    pub sample_cap: usize,
    pub sample_bands: Option<String>,
    /// node budget per move instead of time (no nps calibration needed)
    pub use_nodes: bool,
    /// write each position only once (Bloom filter); otherwise all are written
    /// and deduplicated later (`movegen dedup`)
    pub dedup: bool,
    /// false-positive rate of the run's Bloom filter (unique count / dedup)
    pub bloom_fp: f64,
}

const RESCORE_VALUE_THRESHOLD: i16 = 5000; // don't bother rescoring decisive/TB-range evals

pub fn expand_game_paths(paths: &[&str]) -> std::io::Result<Vec<String>> {
    let mut result = Vec::new();

    for &path in paths {
        let p = Path::new(path);
        if p.is_dir() {
            let mut dir_games: Vec<String> = std::fs::read_dir(p)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|entry_path| {
                    entry_path.is_file()
                        && entry_path
                            .extension()
                            .map_or(false, |ext| ext.eq_ignore_ascii_case("games"))
                })
                .map(|entry_path| entry_path.to_string_lossy().into_owned())
                .collect();
            dir_games.sort(); // deterministic order, independent of readdir order
            result.extend(dir_games);
        } else {
            result.push(path.to_string());
        }
    }

    Ok(result)
}

fn count_positions_to_rescore(paths: &[&str], threshold: i16) -> std::io::Result<u64> {
    let hll = HyperLogLog::<Precision14, 5>::default();
    let mut exact: u64 = 0;
    let paths = expand_game_paths(&paths)?;
    for path in paths.iter() {
        let mut reader = BufReader::new(File::open(path)?);
        for sample in reader.iter_samples() {
            if sample.position.has_capture() {
                continue;
            }
            if sample.position.bp == 0 || sample.position.wp == 0 {
                continue;
            }
            if sample.value.abs() >= 15000 {
                continue;
            }

            // only positions below the threshold actually get rescored.
            // Exact count (the HyperLogLog estimate was badly off, e.g. 164 for
            // 2.5M positions); duplicates are counted too, which is exact for the
            // already deduplicated chunk files.
            if sample.value.abs() < threshold {
                exact += 1;
            }
        }
    }

    let _ = &hll;
    Ok(exact)
}

pub fn rescoring_data(
    paths: Vec<&str>,
    output: &str,
    time: i32,
    num_workers: usize,
    partitions: usize,
    queue_capacity: usize,
) -> std::io::Result<()> {
    println!(
        "Counting positions that need rescoring (threshold={})...",
        RESCORE_VALUE_THRESHOLD
    );
    let count_start = Instant::now();
    let to_rescore = count_positions_to_rescore(&paths, RESCORE_VALUE_THRESHOLD)?;

    let paths = expand_game_paths(&paths)?;
    println!(
        "Found {} positions to rescore (counted in {:.1}s)",
        to_rescore,
        count_start.elapsed().as_secs_f64()
    );

    // rough ETA: each worker handles ~1 position per `time` ms, workers run in parallel
    let estimated_secs = (to_rescore as f64) * (time as f64 / 1000.0) / (num_workers as f64);
    let estimated_hours = estimated_secs / 3600.0;
    println!(
        "Estimated rescoring time: ~{:.1} minutes ({:.2} hours) with {} workers at {}ms/position",
        estimated_secs / 60.0,
        estimated_hours,
        num_workers,
        time
    );

    let mut total_count: u64 = 0;
    let mut skipped_rescoring: u64 = 0;

    let (work_tx, work_rx) = mpsc::sync_channel::<Sample::Sample>(queue_capacity);
    let work_rx = Arc::new(Mutex::new(work_rx));

    let (result_tx, result_rx) = mpsc::sync_channel::<Sample::Sample>(queue_capacity);

    // now we can size the bar properly and get a real ETA display from indicatif
    let bar = ProgressBar::new(to_rescore);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise}] {bar:40.cyan/blue} {pos:>9}/{len:9} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
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
        let worker_bar = bar.clone();
        let handle = thread::spawn(move || {
            let mut command = Command::new("./MainEngine")
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
                worker_bar.inc(1); // only actually-rescored positions advance this bar

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
        for sample in reader.iter_samples() {
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

            if sample.value.abs() >= RESCORE_VALUE_THRESHOLD {
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

    drop(work_tx);
    drop(result_tx);
    for handle in worker_handles {
        handle.join().unwrap();
    }
    writer_handle.join().unwrap();
    bar.finish();

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
    remove_partitions(output, partitions)?;

    Ok(())
}

/// Creates an opening book with the engine's `--book` mode.
///
/// Every line of `input` is a FEN. `num_workers` engine processes are started
/// as `./MainEngine --book <engine_args...>`; each one prints the qualifying
/// positions for a FEN followed by "done". The collected positions are
/// deduplicated, shuffled (the match tool hands every parallel game a
/// contiguous chunk of the book, and neighbouring positions from the same
/// root opening are nearly identical) and optionally cut to `max_positions`.
///
/// Engine options (see main.cpp): --book_plies N, --book_search_depth D,
/// --book_min A, --book_max B, --book_abs 1, --book_hash MB.
pub fn create_book(
    input: &str,
    output: &str,
    num_workers: usize,
    engine_args: &[&str],
    max_positions: Option<usize>,
    seed: u64,
) -> std::io::Result<usize> {
    let (tx, rx): (Sender<String>, Receiver<String>) = mpsc::channel();
    let open_reader = BufReader::new(File::open(input)?);
    let openings: Vec<String> = open_reader
        .lines()
        .map(|value| value.unwrap().trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    let bar = ProgressBar::new(openings.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise},{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let chunk_size = std::cmp::max(1, (openings.len() + num_workers - 1) / num_workers.max(1));
    let mut handles = Vec::new();
    for chunk in openings.chunks(chunk_size) {
        let sender = tx.clone();
        let my_chunk = chunk.to_owned();
        let args: Vec<String> = engine_args.iter().map(|a| a.to_string()).collect();
        handles.push(thread::spawn(move || {
            let mut command = Command::new("./MainEngine")
                .arg("--book")
                .args(&args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("Failed to start ./MainEngine");
            let mut stdin = command.stdin.take().unwrap();
            let stdout = command.stdout.take().unwrap();
            let mut f = BufReader::new(stdout);

            for pos in my_chunk {
                stdin.write_all((pos.clone() + "\n").as_bytes()).unwrap();
                loop {
                    let mut buffer = String::new();
                    match f.read_line(&mut buffer) {
                        Ok(0) => panic!("engine closed its output while generating the book"),
                        Ok(_) => {}
                        Err(e) => panic!("{:?}", e),
                    }
                    let line = buffer.trim().to_string();
                    let done = line == "done";
                    sender.send(line).unwrap();
                    if done {
                        break;
                    }
                }
            }
            let _ = stdin.write_all(b"terminate\n");
            let _ = command.wait();
        }));
    }
    // all clones live in the workers; without this the receiver never ends
    drop(tx);

    let mut seen: HashSet<String> = HashSet::new();
    let mut positions: Vec<String> = Vec::new();
    for val in rx {
        if val == "done" {
            bar.inc(1);
        } else if val.starts_with("B:") || val.starts_with("W:") {
            if seen.insert(val.clone()) {
                positions.push(val);
            }
        }
    }
    for h in handles {
        h.join().expect("book worker panicked");
    }
    bar.finish();

    let mut rng = StdRng::seed_from_u64(seed);
    positions.shuffle(&mut rng);
    if let Some(max) = max_positions {
        positions.truncate(max);
    }

    let mut writer = BufWriter::new(File::create(output)?);
    for p in &positions {
        writer.write_all((p.clone() + "\n").as_bytes())?;
    }
    writer.flush()?;
    println!("{}: {} positions", output, positions.len());
    Ok(positions.len())
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
    writer.flush()?;
    remove_partitions(input, partitions)?;

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
    let paths = expand_game_paths(&paths)?;
    let mut writer = BufWriter::with_capacity(10000, File::create(output)?);
    let mut filter = Bloom::new_for_fp_rate(4000000000, 0.01);
    let mut total_count: u64 = 0;
    let mut written_count: u64 = 0;

    for path in paths {
        let mut reader = BufReader::with_capacity(1000000, File::open(path)?);
        for game in reader.iter_games() {
            let mut counter = 0;

            if game.result == Result::DRAW || game.result == Result::UNKNOWN {
                continue;
            }

            let samples = game.get_samples();
            for (index, sample) in samples.iter().enumerate() {
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
                    copy.mlh = samples.len() as i16 - index as i16;
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

fn keep(s: &Sample::Sample) -> bool {
    let p = s.position;
    if p.bp == 0 || p.wp == 0 {
        return false;
    }
    if s.value.abs() >= 15000 {
        return false;
    }
    // legality
    if (p.bp & p.wp) != 0 {
        return false;
    }
    if (p.k & !(p.bp | p.wp)) != 0 {
        return false;
    }
    if (p.wp & !p.k & 0x0000_000f) != 0 {
        return false;
    }
    if (p.bp & !p.k & 0xf000_0000) != 0 {
        return false;
    }
    if p.bp.count_ones() > 12 || p.wp.count_ones() > 12 {
        return false;
    }
    // side to move must be quiet and have more than one move
    if p.has_capture() {
        return false;
    }
    let mut l = MoveList::new();
    l.get_moves(p);
    l.length > 1
}

pub fn filter_training_data(path: &str, out: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut writer = BufWriter::new(File::create(out)?);
    let sample_iter = reader.iter_samples();

    for s in sample_iter {
        if !keep(&s) {
            continue;
        }

        s.write_fen(&mut writer)?;
    }
    Ok(())
}

pub fn get_unique_samples(
    paths: Vec<&str>,
    output: &str,
    partitions: usize,
    base: &TableBase::Base,
) -> std::io::Result<()> {
    //let mut reader = BufReader::new(File::open(path)?);
    let paths = expand_game_paths(&paths)?;
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
    remove_partitions(output, partitions)?;
    Ok(())
}
pub fn create_policy_data(
    paths: Vec<&str>,
    output: &str,
    partitions: usize,
) -> std::io::Result<()> {
    let paths = expand_game_paths(&paths)?;
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
        let mut reader = BufReader::new(File::open(&path)?);
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
    files.clear(); // flush + close the partition writers before reading them back
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
    remove_partitions(output, partitions)?;

    Ok(())
}

/// Splits unrescored samples (as written by `Generator::generate_games`) into a
/// training- and a validation-file.
///
/// The split is decided by a deterministic hash of the position, not by a random
/// draw, so every copy of a position lands in the same set (no train/val leakage
/// through duplicate positions) and re-running gives the same split.
/// `max_samples` limits how many samples are read from the start of `input`
/// (`None` = whole file). Because the split is per-position and the limit takes a
/// prefix, a larger limit gives train/val sets that contain the smaller ones.
/// Streams the input; memory use is constant.
/// Returns (train_count, val_count).
pub fn split_train_val(
    input: &str,
    train_output: &str,
    val_output: &str,
    val_fraction: f64,
    max_samples: Option<usize>,
) -> std::io::Result<(u64, u64)> {
    use std::collections::hash_map::DefaultHasher; // fixed keys -> deterministic across runs
    use std::hash::Hasher;

    assert!(
        (0.0..=1.0).contains(&val_fraction),
        "val_fraction must be in [0, 1]"
    );
    let threshold = (val_fraction * u64::MAX as f64) as u64;

    let mut reader = BufReader::new(File::open(input)?);
    let mut train_writer = BufWriter::new(File::create(train_output)?);
    let mut val_writer = BufWriter::new(File::create(val_output)?);
    let (mut train_count, mut val_count) = (0u64, 0u64);

    for sample in reader.iter_samples().take(max_samples.unwrap_or(usize::MAX)) {
        let mut hasher = DefaultHasher::new();
        sample.position.hash(&mut hasher);
        if hasher.finish() < threshold {
            sample.write_fen(&mut val_writer)?;
            val_count += 1;
        } else {
            sample.write_fen(&mut train_writer)?;
            train_count += 1;
        }
    }
    train_writer.flush()?;
    val_writer.flush()?;

    println!("Train samples: {train_count}, validation samples: {val_count}");
    Ok((train_count, val_count))
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
            sample_prob: 1.0,
            sample_cap: 2500,
            sample_bands: None,
            use_nodes: false,
            dedup: false,
            bloom_fp: 1e-3,
        }
    }

    pub fn generate_games(&self) -> std::io::Result<()> {
        // sized for this run: max_samples unique positions (+10 % margin) at
        // false-positive rate bloom_fp (a false positive drops a unique position
        // when dedup is on). ~14.4 bits per entry at 1e-3: 3e9 positions ~6 GB.
        let bloom_items = ((self.max_samples as f64) * 1.1).max(1e6) as usize;
        let mut filter = Bloom::new_for_fp_rate(bloom_items, self.bloom_fp);
        println!(
            "bloom filter: {} entries at fp {} (~{:.2} GB)",
            bloom_items, self.bloom_fp,
            bloom_items as f64 * -self.bloom_fp.ln() / (2f64.ln().powi(2)) / 8e9
        );
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
        // bounded: with hundreds of workers a slow writer must block them, not buffer
        // an unlimited number of games in memory
        let (tx, rx) = mpsc::sync_channel::<Vec<String>>(4 * self.num_workers.max(1));
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
        // search budget per move: node budget (--nodes, no time limit) or time
        let budget = if self.use_nodes {
            format!("--nodes {}", max_nodes)
        } else {
            format!("--time {}", time)
        };
        let mut sampling = format!("--sample_prob {} --sample_cap {}", self.sample_prob, self.sample_cap);
        if let Some(b) = &self.sample_bands {
            sampling += &format!(" --sample_bands {}", b);
        }
        let (mut written, mut skipped) = (0u64, 0u64);
        for id in 0..self.num_workers {
            let budget = budget.clone();
            let sampling = sampling.clone();
            let open = Arc::clone(&openings);
            let op_counter = Arc::clone(&opening_counter);
            let sender = tx.clone();
            let counter = Arc::clone(&thread_counter);
            let worker_seed: u64 =
                rand::thread_rng().gen::<u64>() ^ (id as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let handle = std::thread::spawn(move || {
                let engine_args = format!(
                        "--generate {} --depth {} --seed {} {}
                         --adj_draw_count 8
                         --adj_draw_score 5
                         --adj_draw_min_ply 5
                         --adj_draw_max_pieces 10
                         --adj_draw_prob 0.9
                         --multi-pv-prob 0.9 
                         --multi-pv-eval-diff 65 
                         --multi-pv-min-pieces 10
                        ",
                        budget, depth, worker_seed, sampling
                    );
                let mut command = Command::new("./MainEngine")
                    .args(engine_args.split_whitespace())
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
                    if stdin.write_all((String::from(trimmed) + "\n").as_bytes()).is_err() {
                        eprintln!("worker {id}: engine not accepting input, stopping");
                        break 'outer;
                    }
                    let mut game = Vec::new();
                    loop {
                        let mut buffer = String::new();
                        match f.read_line(&mut buffer) {
                            // engine died: stop this worker (an EOF loop here used to
                            // fill memory with empty lines until the machine crashed)
                            Ok(0) => {
                                eprintln!("worker {id}: engine closed its output, stopping");
                                break 'outer;
                            }
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("worker {id}: read error {:?}, stopping", e);
                                break 'outer;
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
                let _ = stdin.write_all((String::from("terminate\n")).as_bytes());
                let _ = command.kill();
            });
            handles.push(handle);
        }
        // only the workers hold senders now: if they all stop, the loop below ends
        drop(tx);
        'game: for game in rx {
            // (sample, first occurrence in the run) - `filter` decides both the
            // unique count and, with dedup, what is written
            let mut reservoir: Vec<(Sample::Sample, bool)> = Vec::new();
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
                //writing the samples to our file
                let mut sample = Sample::Sample::default();
                sample.position = position;
                sample.result = Sample::Result::from(result_string.as_str());

                /*
                if sample.result == Sample::Result::UNKNOWN {
                    println!("Error UNKNOWN result");
                    println!("{:?}", game.first().unwrap());
                    continue 'game;
                }
                */
                total_count += 1;
                sample.position = if sample.position.color == -1 {
                    sample.position.get_color_flip()
                } else {
                    sample.position
                };
                let is_new = !filter.check(&sample.position) && !sample.position.has_capture();
                reservoir.push((sample, is_new));

                if is_new {
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
            // write the game's samples: quiet positions with pieces of both
            // colours only (the engine already samples quiet positions only)
            for (sample, is_new) in reservoir {
                let p = &sample.position;
                if p.has_capture() || p.wp == 0 || p.bp == 0 {
                    skipped += 1;
                    continue;
                }
                if self.dedup && !is_new {
                    skipped += 1;
                    continue;
                }
                sample
                    .write_fen(&mut writer)
                    .expect("Could not write the data");
                written += 1;
            }
        }

        for handle in handles {
            handle.join().unwrap();
        }
        println!(
            "We got back {} unique samples and a total of {}; written {}, skipped {}{}",
            unique_count, total_count, written, skipped,
            if self.dedup { " (dedup on)" } else { "" }
        );
        writer.flush().expect("Could not flush writer");

        Ok(())
    }
}

/// One-off exact-ish dedup of a sample file before chunked rescoring.
/// Same filters as `rescoring_children` (no capture, both colours present,
/// |value| < 15000); a Bloom filter sized for `expected` positions with
/// false-positive rate `fp` (1e-4 for ~400M entries is about 1 GB).
pub fn dedup_samples(input: &str, output: &str, expected: usize, fp: f64) -> std::io::Result<(u64, u64)> {
    let mut filter = Bloom::new_for_fp_rate(expected, fp);
    let mut reader = BufReader::with_capacity(1 << 22, File::open(input)?);
    let mut writer = BufWriter::with_capacity(1 << 22, File::create(output)?);
    let (mut read, mut written) = (0u64, 0u64);
    for sample in reader.iter_samples() {
        read += 1;
        if read % 10_000_000 == 0 {
            println!("dedup: {} read, {} written", read, written);
        }
        if sample.position.has_capture()
            || sample.position.bp == 0
            || sample.position.wp == 0
            || sample.value.abs() >= 15000
        {
            continue;
        }
        if filter.check(&sample.position) {
            continue;
        }
        filter.set(&sample.position);
        sample.write_fen(&mut writer)?;
        written += 1;
    }
    writer.flush()?;
    println!("dedup done: {} read, {} written to {}", read, written, output);
    Ok((read, written))
}

/// Splits a sample file into `chunks` contiguous files `<prefix>c<i>.samples`
/// (i = 1..chunks) with (almost) equal counts.
pub fn split_into_chunks(input: &str, prefix: &str, chunks: usize) -> std::io::Result<Vec<u64>> {
    const SAMPLE_BYTES: u64 = 18;
    let total = std::fs::metadata(input)?.len() / SAMPLE_BYTES;
    let mut reader = BufReader::with_capacity(1 << 22, File::open(input)?);
    let mut iter = reader.iter_samples();
    let mut counts = Vec::new();
    for i in 0..chunks as u64 {
        let target = total * (i + 1) / chunks as u64 - total * i / chunks as u64;
        let name = format!("{}c{}.samples", prefix, i + 1);
        let mut writer = BufWriter::with_capacity(1 << 22, File::create(&name)?);
        let mut n = 0u64;
        while n < target {
            match iter.next() {
                Some(s) => {
                    s.write_fen(&mut writer)?;
                    n += 1;
                }
                None => break,
            }
        }
        writer.flush()?;
        println!("{}: {} samples", name, n);
        counts.push(n);
    }
    Ok(counts)
}

/// Like `split_into_chunks`, but every sample goes to a uniformly random chunk:
/// each chunk is a random sample of the whole input (no drift along the file, e.g.
/// from dedup keeping first occurrences). Chunk sizes vary by about sqrt(n).
pub fn split_into_random_chunks(input: &str, prefix: &str, chunks: usize, seed: u64) -> std::io::Result<Vec<u64>> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut reader = BufReader::with_capacity(1 << 22, File::open(input)?);
    let mut writers = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let name = format!("{}c{}.samples", prefix, i + 1);
        writers.push(BufWriter::with_capacity(1 << 20, File::create(&name)?));
    }
    let mut counts = vec![0u64; chunks];
    for s in reader.iter_samples() {
        let c = rng.gen_range(0..chunks);
        s.write_fen(&mut writers[c])?;
        counts[c] += 1;
    }
    for (i, w) in writers.iter_mut().enumerate() {
        w.flush()?;
        println!("{}c{}.samples: {} samples", prefix, i + 1, counts[i]);
    }
    Ok(counts)
}

/// External shuffle of a POL1 policy-sample file (variable-length records):
/// records go to `partitions` random temp files, each partition is shuffled
/// in RAM and appended to the output (new header with the same count).
pub fn shuffle_policy_external(input: &str, output: &str, partitions: usize) -> std::io::Result<u64> {
    use crate::Sample::{read_policy_header, write_policy_header, PolicySample};
    let mut reader = BufReader::with_capacity(1 << 22, File::open(input)?);
    let count = read_policy_header(&mut reader)?;
    let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    let names: Vec<String> = (0..partitions).map(|i| format!("{}.part{}", output, i)).collect();
    {
        let mut parts: Vec<BufWriter<File>> = names
            .iter()
            .map(|n| File::create(n).map(|f| BufWriter::with_capacity(1 << 20, f)))
            .collect::<std::io::Result<_>>()?;
        let mut ps = PolicySample::new(Sample::Sample::default());
        for i in 0..count {
            ps.read_into(&mut reader)?;
            ps.write(&mut parts[rng.gen_range(0..partitions)])?;
            if (i + 1) % 10_000_000 == 0 {
                println!("pshuffle: distributed {} / {}", i + 1, count);
            }
        }
        for p in parts.iter_mut() {
            p.flush()?;
        }
    }
    let mut writer = BufWriter::with_capacity(1 << 22, File::create(output)?);
    write_policy_header(&mut writer, count)?;
    let mut written = 0u64;
    for name in names.iter() {
        let mut r = BufReader::with_capacity(1 << 22, File::open(name)?);
        let mut buf: Vec<PolicySample> = Vec::new();
        while r.has_data_left()? {
            let mut ps = PolicySample::new(Sample::Sample::default());
            ps.read_into(&mut r)?;
            buf.push(ps);
        }
        buf.shuffle(&mut rng);
        for ps in buf.iter() {
            ps.write(&mut writer)?;
        }
        written += buf.len() as u64;
        std::fs::remove_file(name)?;
    }
    writer.flush()?;
    println!("pshuffle done: {} of {} samples written to {}", written, count, output);
    Ok(written)
}

