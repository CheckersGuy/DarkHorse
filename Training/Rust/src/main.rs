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
use Data::count_unique_samples;
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

    /*
        Data::remove_samples(
            "/mnt/e/newtry11rescoredmlhshuffledx.samples",
            "/mnt/e/validation.samples",
            "/mnt/e/master1.samples",
        )?;
    */
    /*let mut generator = Generator::new(
        String::from("../Positions/ultrabook2.pos"),
        String::from("/mnt/e/evalnexttest.games"),
        14,
        400000000,
    );

    generator.time = 1;
    generator.max_nodes = 250000000;
    generator.depth = 70;
    generator.generate_games()?;

    */

    //Data::create_policy_data("E:\\Iamhere8.samples", "E:\\Iamhere8policy.samples");
    /*let mut writer = BufWriter::new(File::create("test.games")?);
    let mut game = Game::new();
    game.set_start_position(Position::get_start_position());

    let fen_strings = vec![
        "W:W21,22,23,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,9,10,12,15",
        "B:W19,21,22,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,9,10,12,15",
        "W:W19,21,22,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12,14,15",
        "B:W18,19,21,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12,14,15",
        "W:W19,21,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12,15,23",
        "B:W11,19,21,24,25,26,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12",
        "W:W21,24,25,26,28,29,30,31,32:B1,2,3,4,5,6,8,10,12,23",
        "B:W19,21,24,25,28,29,30,31,32:B1,2,3,4,5,6,8,10,12",
    /];

    for fen in fen_strings.iter() {
        let position = Position::try_from(*fen).expect("Could not parse fen_string");
        position.print_position();
        println!();
    }
    for fen in fen_strings.iter() {
        let position = Position::try_from(*fen).expect("Could not parse fen_string");
        let added = game.add_position(position);
        if added.is_none() {
            println!("Could not add the position");
        }
    }
    println!("Reading the positions from the game");

    for position in game.get_positions().iter() {
        position.print_position();
        println!();
    }


    */

    /*Data::create_policy_data(
        vec![
            "/mnt/e/finaldataset0.games",
            "/mnt/e/finaldataset1.games",
            "/mnt/e/finaldataset2.games",
            "/mnt/e/finaldataset3.games",
        ],
        "/mnt/e/finalpolicy.samples",
    )?;
    */
    /*let base = Base::new("/mnt/c/kr_english_wld", 2000, 10).unwrap();
    Data::rescore_games(
        vec![
            "/mnt/c/TrainData/finaldataset0.games",
            "/mnt/c/TrainData/finaldataset1.games",
            "/mnt/c/TrainData/finaldataset2.games",
            "/mnt/c/TrainData/finaldataset3.games",
        ],
        "/mnt/c/TrainData/value.samples",
        &base,
        32,
    )?;
    */

    Data::filter_training_data(
        "c:/TrainData/value.samples",
        "c:/TrainData/valuefiltered.samples",
    )?;

    Ok(())
}
