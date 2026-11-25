#![feature(buf_read_has_data_left)]
#![feature(iter_next_chunk)]
#![feature(iter_array_chunks)]
#![feature(core_intrinsics)]
pub mod Data;
pub mod Histogram;
pub mod Pos;
pub mod Sample;
pub mod TableBase;
pub mod dataloader;
use anyhow::Context;
use bloomfilter::reexports::bit_vec::BitBlock;
use itertools::Itertools;
use std::fs::File;
use std::intrinsics::size_of;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::iter::zip;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::sync::mpsc;
use std::time::Instant;
use std::usize;
use Data::Generator;
use Pos::Square;
use Pos::*;
use Sample::Game;
use Sample::Result;
use Sample::SampleIteratorTrait;
use Sample::SampleType;
use TableBase::Base;

use crate::Sample::OldSample;

fn main() -> anyhow::Result<()> {
    // Data steps for the chunked policy rescoring (one call per step):
    //   movegen dedup   <in> <out> [expected] [fp]  (Bloom filter sized for the input file,
    //                                               fp 1e-4 by default)
    //   movegen split   <in> <train_out> <val_out> <val_fraction>
    //   movegen chunks  <in> <prefix> <n> [random [seed]] -> <prefix>c1..cN.samples
    //                                               (random: each sample to a random chunk)
    //   movegen rescore <in> <out> <workers> [A,B,C,D]  (children, ./MainEngine; node budget
    //                                                per search by pieces >=17/13-16/9-12/<=8,
    //                                                default Data::DEFAULT_NODES_PHASE;
    //                                                root value without MLH, children with)
    //   movegen pshuffle <in> <out>                 (external shuffle of a POL1 policy file)
    //   movegen vrescore <in> <out> <workers> [A,B,C,D] (value net: root value only, same
    //                                                budget and root search; POL1, no children)
    //   movegen book <in> <out> <workers> <plies> [min max] [max_positions]
    //                                               opening book: positions <plies> deeper, depth-1
    //                                               score in [min,max] (default -55..55)
    //   movegen generate <book> <out> <workers> <max_samples> [key=value ...]
    //                                               self-play games with ./MainEngine --generate,
    //                                               samples from the search trees. Keys:
    //                                               nodes=N (node budget per move, no time limit)
    //                                               time=MS (default 1, used without nodes=)
    //                                               prob=P (keep prob. per new position, default 1)
    //                                               cap=N (max positions per game, default 2500)
    //                                               bands=a,b,c,d (keep factor by |eval| band
    //                                               <=50/51-200/201-400/>400, default all 1)
    //                                               dedup=1 (write each position once)
    //                                               depth=D (default 128)
    //                                               bloom_fp=P (dedup filter fp, default 1e-3;
    //                                               filter sized max_samples * 1.1)
    raise_open_file_limit();
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "dedup" => {
                // Bloom filter sized for the input (18 bytes per sample) unless given:
                // movegen dedup <in> <out> [expected_positions] [fp], default fp 1e-4
                // (~19 bits per entry: 3e9 positions ~7.2 GB)
                let expected: usize = if args.len() > 4 {
                    args[4].parse()?
                } else {
                    ((std::fs::metadata(&args[2])?.len() / 18) as usize).max(1_000_000)
                };
                let fp: f64 = if args.len() > 5 { args[5].parse()? } else { 1e-4 };
                println!(
                    "dedup bloom filter: {} entries at fp {} (~{:.2} GB)",
                    expected, fp,
                    expected as f64 * -fp.ln() / (2f64.ln().powi(2)) / 8e9
                );
                Data::dedup_samples(&args[2], &args[3], expected, fp)?;
            }
            "split" => {
                let frac: f64 = args[5].parse()?;
                let (t, v) = Data::split_train_val(&args[2], &args[3], &args[4], frac, None)?;
                println!("split done: {} train, {} val", t, v);
            }
            "chunks" => {
                let n: usize = args[4].parse()?;
                // movegen chunks <in> <prefix> <n> [random [seed]]: random = every sample
                // goes to a random chunk (each chunk is a random sample of the input)
                if args.len() > 5 && args[5] == "random" {
                    let seed: u64 = if args.len() > 6 { args[6].parse()? } else { 0x5EED };
                    Data::split_into_random_chunks(&args[2], &args[3], n, seed)?;
                } else {
                    Data::split_into_chunks(&args[2], &args[3], n)?;
                }
            }
            "rescore" => {
                let workers: usize = args[4].parse()?;
                let np = if args.len() > 5 { args[5].as_str() } else { Data::DEFAULT_NODES_PHASE };
                Data::rescoring_children_nodes(vec![args[2].as_str()], &args[3], np, workers, 32, 100000)?;
            }
            "vrescore" => {
                let workers: usize = args[4].parse()?;
                let np = if args.len() > 5 { args[5].as_str() } else { Data::DEFAULT_NODES_PHASE };
                Data::rescoring_value(vec![args[2].as_str()], &args[3], np, workers, 32, 100000)?;
            }
            "generate" => {
                let workers: usize = args[4].parse()?;
                let max_samples: usize = args[5].parse()?;
                let mut generator = Generator::new(args[2].clone(), args[3].clone(), workers, max_samples);
                generator.time = 1;
                for kv in &args[6..] {
                    let (key, val) = kv.split_once('=').with_context(|| format!("expected key=value, got {kv}"))?;
                    match key {
                        "nodes" => {
                            generator.max_nodes = val.parse()?;
                            generator.use_nodes = true;
                        }
                        "time" => generator.time = val.parse()?,
                        "prob" => generator.sample_prob = val.parse()?,
                        "cap" => generator.sample_cap = val.parse()?,
                        "bands" => {
                            anyhow::ensure!(val.split(',').count() == 4, "bands needs 4 values");
                            generator.sample_bands = Some(val.to_string());
                        }
                        "dedup" => generator.dedup = val == "1" || val == "true",
                        "depth" => generator.depth = val.parse()?,
                        "bloom_fp" => generator.bloom_fp = val.parse()?,
                        other => anyhow::bail!("unknown generate option {other}"),
                    }
                }
                generator.generate_games()?;
            }
            "book" => {
                // movegen book <in> <out> <workers> <plies> [min max] [max_positions]:
                // every input position expanded <plies> plies deeper; keeps leaves whose
                // depth-1 score lies in [min, max] (default -55..55), deduplicated and
                // shuffled (./MainEngine --book)
                let workers: usize = args[4].parse()?;
                let plies = args[5].clone();
                let (lo, hi) = if args.len() > 7 { (args[6].clone(), args[7].clone()) } else { ("-55".to_string(), "55".to_string()) };
                let max: Option<usize> = if args.len() > 8 { Some(args[8].parse()?) } else { None };
                let engine_args = ["--book_plies", plies.as_str(), "--book_search_depth", "1",
                                   "--book_min", lo.as_str(), "--book_max", hi.as_str()];
                Data::create_book(&args[2], &args[3], workers, &engine_args, max, 1)?;
            }
            "pshuffle" => {
                Data::shuffle_policy_external(&args[2], &args[3], 64)?;
            }
            other => anyhow::bail!("unknown command {other}"),
        }
        return Ok(());
    }
    /*let mut generator = Generator::new(
        String::from("../Positions/tiny.book"),
        String::from("./goodlord2.samples"),
        14,
        5000000000,
    );

    generator.time = 1;
    generator.max_nodes = 250000000;
    generator.depth = 100;
    generator.generate_games()?;
    */
    /*let path = "/mnt/c/TrainingData/goodlord2.rescored.samples";
    let mut reader = BufReader::new(File::open(path)?);

    let mut histogramm = Histogram::Histogram::default();
    for sample in reader.iter_samples() {
        let wp = sample.position.wp;
        let bp = sample.position.bp;
        histogramm.add(wp, bp);
    }

    histogramm.print();
    */

    //Data::create_book("../Positions/drawbook.book", "../Positions/tiny.book", 14, &[], None, 1).expect("Failed");

    /*
        Data::get_unique_samples(
            vec![
                "D:/TrainData/Games/training-book.games",
                "D:/TrainData/Games/training-book2.games",
                "D:/TrainData/Games/training-book3.games",
                "D:/TrainData/Games/multipv.games",
            ],
            "D:/TrainData/Samples/training-book.samples",
            32,
        )?;
    */
    /*
        Data::create_mlh_data(
            vec!["/mnt/c/TrainingData/finaldataset0.games"],
            "/mnt/c/TrainingData/mlh-data.samples",
        )?;
    */
    /*Data::filter_training_data(
        "/mnt/c/TrainingData/reservoir-test.rescored.samples",
        "/mnt/c/TrainingData/reservoir-test.rescored.filtered.samples",
    )?;
    */

    /*Data::rescoring_data(
        vec![
            "/mnt/d/TrainData/Games/training-book.games",
            "/mnt/d/TrainData/Games/training-book2.games",
            "/mnt/d/TrainData/Games/training-book3.games",
            "/mnt/d/TrainData/Games/cloud.games",
            "/mnt/d/TrainData/Games/cloud2.games",
            "/mnt/d/TrainData/Games/cloud3.games",
            "/mnt/d/TrainData/Games/cloud4.games",
            "/mnt/d/TrainData/Games/debugging0.games",
            "/mnt/d/TrainData/Games/debugging1.games",
            "/mnt/d/TrainData/Games/multipv.games",
            "/mnt/d/TrainData/Games/multipv2.games",
            "/mnt/d/TrainData/Games/qs_samples0.games",
        ],
        "/mnt/d/TrainData/Samples/rescored.samples",
        2,
        14,
        16,
        1000000,
    )?;
    */
    /*Data::rescoring_children(
        vec!["/mnt/c/TrainingData/reservoir-test.samples"],
        "/mnt/c/TrainingData/test-samples.children.samples",
        1,
        1,
        14,
        32,
        100000,
    )?;
    */

    // 10M positions total, ~80/20 train/val split (already done)
    /*Data::split_train_val(
        "/mnt/c/TrainingData/reservoir-test.samples",
        "/mnt/c/TrainingData/reservoir-test.10M.train.samples",
        "/mnt/c/TrainingData/reservoir-test.10M.val.samples",
        0.2,
        Some(10_000_000),
    )?;
    */

    // rescore both sets with children, then delete the unrescored inputs (done)
    /*
    for set in ["train", "val"] {
        let input = format!("/mnt/c/TrainingData/reservoir-test.10M.{set}.samples");
        let output = format!("/mnt/c/TrainingData/reservoir-test.10M.{set}.children.samples");
        println!("Rescoring {input}");
        Data::rescoring_children(vec![input.as_str()], &output, 1, 1, 14, 32, 100000)?;
    }
    for set in ["train", "val"] {
        std::fs::remove_file(format!("/mnt/c/TrainingData/reservoir-test.10M.{set}.samples"))?;
    }
    */

    /* (done) // Unbalanced opening books for SPRT/SPSA (needs ./MainEngine in this dir).
    // Stage 1: positions 6 plies after the drawbook openings whose quick
    // (depth 1) score is moderately unbalanced for either side.
    let candidates = "../Positions/unbalanced.candidates";
    Data::create_book(
        "../Positions/drawbook.book",
        candidates,
        14,
        &["--book_plies", "6", "--book_search_depth", "1", "--book_abs", "1",
          "--book_min", "30", "--book_max", "300"],
        Some(200_000),
        1,
    )?;
    // Stage 2: rescore the candidates with a depth-12 search and keep the ones
    // inside each |score| window.
    for (lo, hi) in [(40, 100), (70, 150), (100, 220)] {
        let (lo_s, hi_s) = (lo.to_string(), hi.to_string());
        Data::create_book(
            candidates,
            &format!("../Positions/unbalanced_{lo}_{hi}.book"),
            14,
            &["--book_plies", "0", "--book_search_depth", "12", "--book_abs", "1",
              "--book_min", lo_s.as_str(), "--book_max", hi_s.as_str(), "--book_hash", "16"],
            None,
            2,
        )?;
    }
    */

    /*
        let mut reader = BufReader::new(File::open("D:/TrainData/windows.samples")?);

        for sample in reader.iter_samples().take(1000) {
            if sample.position.color == -1 {
                println!("Wrong sample in dataset");
            }
        }
    */
    Ok(())
}

/// Every engine worker holds two pipe fds (rescore/generate); hundreds of workers
/// on a cloud machine can exceed the usual soft limit of 1024 open files.
fn raise_open_file_limit() {
    unsafe {
        let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) == 0 && rl.rlim_cur < rl.rlim_max {
            rl.rlim_cur = rl.rlim_max;
            libc::setrlimit(libc::RLIMIT_NOFILE, &rl);
        }
    }
}
