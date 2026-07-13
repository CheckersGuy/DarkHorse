#![feature(buf_read_has_data_left)]
#![feature(iter_next_chunk)]
#![feature(iter_array_chunks)]
#![feature(core_intrinsics)]
pub mod Data;
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
    println!("Starting process");
    /*let mut generator = Generator::new(
        String::from("../Positions/training2.book"),
        String::from("./cloud.games"),
        14,
        4000000000,
    );

    generator.time = 1;
    generator.max_nodes = 250000000;
    generator.depth = 70;
    generator.generate_games()?;
    */

    /*
        Data::create_book(
            "C:/Users/leagu/DarkHorse/Training/Positions/drawbook.book",
            "C:/Users/leagu/DarkHorse/Training/Positions/training2.book",
            14,
        )
        .expect("Failed");

    */
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
            vec![
                "/mnt/d/TrainData/Games/training-book.games",
                "/mnt/d/TrainData/Games/training-book2.games",
                "/mnt/d/TrainData/Games/training-book3.games",
            ],
            "/mnt/d/TrainData/Samples/mlh.samples",
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
    Data::rescoring_data(
        vec!["/mnt/d/TrainData/Games/debugging0.games"],
        "/mnt/d/TrainData/Samples/debugging-rescored20ms.samples",
        20,
        14,
        4,
        1000000,
    )?;

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
