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
use Data::{count_unique_samples, create_unique_fens, Generator, Rescorer};
use Pos::Position;
use Sample::SampleType;
use Sample::{Result, SampleIteratorTrait};
fn main() -> std::io::Result<()> {
    /*let mut dataloader =
        DataLoader::new(String::from("../TrainData/test.samples"), 1000000, false)?;

    for _ in 0..3000 {
        let sample = dataloader.get_next()?;
        if let SampleType::Fen(ref position) = sample.position {
            let pos = Position::try_from(position.as_str())?;
            pos.print_position();

            println!("Result is is: {:?} and fen: {}", sample.result, position);
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
            vec![
                "../TrainData/newopen12.samples",
                "../TrainData/newopen13.samples",
            ],
            "../TrainData/merged2.samples",
        )?;
    */
    //let distribution = Data::material_distrib("../TrainData/testing2.samples")?;
    //println!("{:?}", distribution);

    //need a new opening book tomorrow
    //soooo much not fun....
    /*let generator = Generator::new(
        String::from("../Positions/newopen2.pos"),
        String::from("../TrainData/newopen8.samples"),
        14,
        4000000,
    );
    */

    /*
        let squares = SampleType::Fen(String::from("B:WK19:BK15,K18"))
            .get_squares()
            .unwrap();

        for square in squares {
            println!("{:?}", square);
        }
    */
    //generator.generate_games()?;

    /* let pos = Pos::Position::try_from("B:W20,21,22,24,25,26,27,30,31,32:B2,3,4,7,8,10,11,12,14,15")
            .unwrap();
        pos.print_position();
    */
    //Data::create_unique_fens("newopen2.pos", "newopen3.pos").unwrap();
    // Data::create_book("../Positions/drawbook.book", "newopen3.pos", 14)?;
    //
    let num_samples = count_unique_samples("../TrainData/test.samples").unwrap();
    println!("{}", num_samples);

    Ok(())
}
