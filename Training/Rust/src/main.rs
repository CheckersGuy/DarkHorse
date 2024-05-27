#![feature(buf_read_has_data_left)]
#![feature(iter_next_chunk)]
pub mod Data;
pub mod Pos;
pub mod Sample;
pub mod TableBase;
pub mod dataloader;
use anyhow::Context;
use arrayvec::ArrayVec;
use itertools::Itertools;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;
use std::usize;
use Data::count_unique_samples;
use Data::Generator;
use Pos::Square;
use Pos::*;
use Sample::SampleIteratorTrait;
use Sample::SampleType;
use TableBase::Base;
pub fn perft(pos: Position, depth: i32) -> usize {
    let mut liste = MoveList::new();
    liste.get_moves(pos);
    if depth == 0 {
        return 1;
    }
    let mut counter: usize = 0;
    for m in liste.iter().dedup() {
        let mut copy_pos = pos.clone();
        copy_pos.make_move(m);
        counter += perft(copy_pos, depth - 1);
    }
    return counter;
}

fn main() -> anyhow::Result<()> {
    /*let mut reader = BufReader::new(File::open(
        "/mnt/e/crazyms1ultimaterescoredshuffled.samples",
    )?);
    //removing call captures
    let mut writer = BufWriter::new(File::create(
        "/mnt/e/crazyms1ultimaterescoredshuffledfixed.samples",
    )?);

    for sample in reader.iter_samples() {
        if let Sample::SampleType::Fen(ref position) = sample.position {
            let pos = Pos::Position::try_from(position.as_str())?;
            if pos.get_jumpers::<1>() == 0 {
                sample.write_fen(&mut writer)?;
            }
        }
    }
    */
    /*
        Data::remove_samples(
            "/mnt/e/newtry11rescoredmlhshuffledx.samples",
            "/mnt/e/validation.samples",
            "/mnt/e/master1.samples",
        )?;
    */
    //Data::create_unique_fens("training.pos", "unique.pos")?;

    //Need to write some code to combine 2 or more sample files
    //which should be straight forward to add
    //
    /* Data::merge_samples(
        vec![
            "../TrainData/newopen14.samples",
            "../TrainData/merged.samples",
        ],
        "../TrainData/merged2.samples",
    )?;
    */
    /*let mut generator = Generator::new(
            String::from("../Positions/ultrabook2.pos"),
            String::from("crazyms1next4.samples"),
            14,
            200000000,
        );
    */
    //generator.time = 1;
    //generator.prev_file = Some("/mnt/e/crazyms1next3.samples");
    // generator.generate_games()?;

    Data::create_subset(
        "/mnt/e/policyultimateshuffled.samples",
        "/mnt/e/smallpolicy.samples",
        2000000,
    )
    .unwrap();

    //Data::create_book("../Positions/drawbook.book", "ultrabook2.pos", 6)?;

    //let fen_string = "B:W30,29:B4,24";
    // let base = Base::new_dtw("E:\\kr_english_wld", "E:\\kr_english_dtw", 2000, 10).unwrap();

    /*
        let result = base.probe_dtw(fen_string).expect("Could not call function");

        if let Some(mlh_counter) = result {
            println!("{mlh_counter}");
        }
    */
    /*
        let result = base.probe("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        base.print_fen("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        println!("{:?}", result);
    */

    /*Data::create_policy_data(
        "D:/crazyms1nextultimaterescored.samples",
        "E:/policyultimate.samples",
        &base,
    )
    */

    /*Data::dump_mlh_samples(
            "/mnt/e/newtry11rescoredmlhshuffled.samples",
            "/mnt/e/mlhshuffled2.samples",
        )?;

    */

    //Data::create_mlh_data("E:/newtry11rescored.samples", "E:/mlh3.samples", &base)?;
    /* Data::shuffle_data_external::<32>(
            "/mnt/e/policyultimate.samples",
            "/mnt/e/policyultimateshuffled.samples",
        )?;
    */
    Ok(())
}
