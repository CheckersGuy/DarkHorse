#![feature(buf_read_has_data_left)]
#![feature(iter_next_chunk)]
pub mod Data;
pub mod Pos;
pub mod Sample;
pub mod TableBase;
pub mod dataloader;
use anyhow::Context;
use arrayvec::ArrayVec;
use bloomfilter::reexports::bit_vec::BitBlock;
use itertools::Itertools;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::iter::zip;
use std::path::Path;
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

fn main() -> anyhow::Result<()> {
    println!("Starting process");
    //Need to check why get-color-flip is not working as expected
    /*let mut reader = BufReader::new(File::open("/mnt/e/weirdstuff.samples")?);

        for sample in reader.iter_samples().take(1000) {
            sample.position.print_position();
            println!("Result: {:?}", sample.result);
            println!();
        }

        let next = Position::get_start_position();
        next.print_position();
        println!();
        next.get_color_flip().print_position();
    */
    /*
        Data::remove_samples(
            "/mnt/e/newtry11rescoredmlhshuffledx.samples",
            "/mnt/e/validation.samples",
            "/mnt/e/master1.samples",
        )?;
    */
    /*let mut generator = Generator::new(
        String::from("../Positions/ultrabook2.pos"),
        String::from("/mnt/e/evalfilter.games"),
        14,
        100000000,
    );

    generator.time = 1;
    generator.max_nodes = 250000000;
    generator.depth = 70;

    generator.generate_games()?;
    */

    //generator.prev_file = Some("/mnt/e/finalrescored/paritysuperiorityshuffled.samples");

    /*Data::create_subset(
            "/mnt/e/policyultimateshuffled.samples",
            "/mnt/e/smallpolicy.samples",
            2000000,
        )
        .unwrap();
    */
    //Data::create_book("../Positions/drawbook.book", "differentbook2.pos", 10)?;

    //let fen_string = "B:W30,29:B4,24";
    let base = Base::new("E:\\kr_english_wld", 2000, 10).unwrap();

    //W:W5:BK32,K31
    //
    //Data::create_mlh_data("E:/Iamhere7.samples", "E:/mlh4.samples", &base)?;
    /*Data::dump_mlh_samples(
            "/mnt/e/newtry11rescoredmlhshuffled.samples",
            "/mnt/e/mlhshuffled2.samples",
        )?;

    */
    //Data::create_mlh_data("E:\\Iamhere8.samples", "E:\\mlh7.samples", &base).expect("Error");
    /*Data::shuffle_data_external::<16>(
        "/mnt/e/coud1.rescored.samples",
        "/mnt/e/coud1.rescored.shuffled.samples",
    )
    .expect("Could not shuffle the training data");*/

    /* Data::merge_rescored_data(
         vec![
             "/mnt/e/coud1.rescored.shuffled.samples",
             "/mnt/e/nextformattest.rescored.shuffled.samples",
         ],
         "/mnt/e/coud2.rescored.shuffled",
     )?;
    */
    /*Data::shuffle_data_external::<16>(
        "E:\\Iamhere9rescored.samples",
        "E:\\Iamhere9shuffled.samples",
    )?;
    */

    //Data::rescore_games("E:\\cloud1.games", "E:\\coud1.rescored.samples", &base)?;

    /*
        let result = base
            .get_move_encoding(
                "B:W9,15,K18,19,21,30:B20,K32",
                "W:W9,15,K18,19,21,30:B20,K27",
            )
            .unwrap();
        println!("Encoding: {}", result);
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

    // Data::convert_samples_to_games("/mnt/e/Iamhere8.samples", "/mnt/e/Iamhere8.games");
    /*Data::filter_training_data(
        "/mnt/e/coud1.rescored.shuffled.samples",
        "/mnt/e/filtereddata.samples",
    )?;*/
    Data::rescore_games(
        "E:\\evalcloudbig.games",
        "E:\\evalcloudbig.rescored.samples",
        &base,
    )?;

    /* let fen_strings = vec![
         "W:W21,22,23,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,9,10,12,15",
         "B:W19,21,22,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,9,10,12,15",
         "W:W19,21,22,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12,14,15",
         "B:W18,19,21,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12,14,15",
         "W:W19,21,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12,15,23",
         "B:W11,19,21,24,25,26,28,29,30,31,32:B1,2,3,4,5,6,7,8,10,12",
         "W:W21,24,25,26,28,29,30,31,32:B1,2,3,4,5,6,8,10,12,23",
         "B:W19,21,24,25,28,29,30,31,32:B1,2,3,4,5,6,8,10,12",
     ];
     let mut pos = Position::get_start_position();
     let mut game = Game::new();
     let mut delta = -20;
     game.set_start_position(pos, delta);
     for (index, fen) in fen_strings.iter().enumerate() {
         delta = delta + (index as i16);
         delta = -delta;
         let position = Position::try_from(*fen).expect("Could not parse fen_string");
         let added = game.add_position(position, delta);
         if added.is_none() {
             println!("Could not add the position");
             break;
         }
     }

     for sample in game.get_samples().iter() {
         println!("-------------------");
         sample.position.print_position();
         println!("{:?}", sample);
     }
    */
    //Data::print_samples("/mnt/e/evalfilter.games")?;

    Ok(())
}
