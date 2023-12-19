#![feature(buf_read_has_data_left)]
pub mod Data;
pub mod Pos;
pub mod Sample;
pub mod dataloader;
use byteorder::{LittleEndian, ReadBytesExt};
use dataloader::DataLoader;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use mktemp::Temp;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::AtomicU32;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc, Mutex};
use std::thread;
use std::{io, path::Path};
use Data::{create_unique_fens, Generator, Rescorer};
use Pos::Position;
use Sample::Result;
use Sample::SampleType;
fn main() -> std::io::Result<()> {
    /*
        let mut dataloader = DataLoader::new(String::from("out.samples"), 1000000, false)?;

        for _ in 0..3000 {
            let sample = dataloader.get_next()?;
            if let SampleType::Fen(ref position) = sample.position {
                let pos = Position::try_from(position.as_str())?;
                pos.print_position();
                let sample_string = match sample.result {
                    Result::LOSS => "LOSS",
                    Result::WIN => "WIN",
                    Result::DRAW => "DRAW",
                    Result::UNKNOWN => "UNKNOWN",
                };
                println!("Result is is: {} and fen: {}", sample_string, position);
            }
            println!();
            println!();
        }
    */

    //Data::create_unique_fens("training.pos", "unique.pos")?;

    //Need to write some code to combine 2 or more sample files
    //which should be straight forward to add
    //
    /*
    Data::merge_samples(
        vec!["../TrainData/test2.samples", "../TrainData/test3.samples"],
        "../TrainData/merged.samples",
    )?;
    */

    /*
    let generator = Generator::new(
        String::from("../Positions/unique.pos"),
        String::from("../TrainData/testing.samples"),
        14,
        5000000,
    );
    generator.generate_games()?;
    */

    Ok(())
}
