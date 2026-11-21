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

    //Data::create_book("../Positions/drawbook.book", "../Positions/tiny.book", 14).expect("Failed");

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

    // rescore both sets with children, then delete the unrescored inputs
    for set in ["train", "val"] {
        let input = format!("/mnt/c/TrainingData/reservoir-test.10M.{set}.samples");
        let output = format!("/mnt/c/TrainingData/reservoir-test.10M.{set}.children.samples");
        println!("Rescoring {input}");
        Data::rescoring_children(vec![input.as_str()], &output, 1, 1, 14, 32, 100000)?;
    }
    for set in ["train", "val"] {
        std::fs::remove_file(format!("/mnt/c/TrainingData/reservoir-test.10M.{set}.samples"))?;
    }

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
