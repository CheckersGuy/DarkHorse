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
        String::from("/mnt/e/Iamhere8.samples"),
        14,
        2000000000,
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

    //Data::create_mlh_data("E:/Iamhere7.samples", "E:/mlh4.samples", &base)?;
    /*Data::dump_mlh_samples(
            "/mnt/e/newtry11rescoredmlhshuffled.samples",
            "/mnt/e/mlhshuffled2.samples",
        )?;

    */
    //Data::create_mlh_data("E:\\Iamhere8.samples", "E:\\mlh7.samples", &base).expect("Error");
    /* Data::shuffle_data_external::<16>(
         "/mnt/e/final1mstestrescored.samples",
         "/mnt/e/final1msshuffled.samples",
     )?;
    */
    /*
            Data::merge_rescored_data(
                vec![
                    "/mnt/e/Iamherenext2rescored.samples",
                    "/mnt/e/Iamhere8rescored.samples",
                ],
                "/mnt/e/Iamherenext3rescored.samples",
            )?;
    */
    /*Data::shuffle_data_external::<16>(
        "E:\\Iamhere9rescored.samples",
        "E:\\Iamhere9shuffled.samples",
    )?;
    */

    /* Data::rescore_games(
        "E:\\Iamhere8.samples",
        "E:\\Iamhere9rescored.samples",
        &base,
    )?;
    */

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
    ];

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
    //iterating over all game in my current dataset
    //and see if the 'game' implementation works
    /*
    let mut game_reader = BufReader::new(File::open("/mnt/e/Iamhere8.samples")?);
    let mut counter: usize = 0;
    //5484 had that particular issue listed below
    //let mut game = game_reader.iter_games().nth(5484).unwrap();
    'outer: for (game_index, game) in game_reader.iter_games().enumerate() {
        let mut test_game = Game::new();

        let pos_iter = game.iter().rev();
        //Looks like there is a data-integrity problem
        //found some positions, that can not belong to the game
        //see below
        //this happens because of they way I am splitting of games from the stream of samples
        //the next starting position just happend to have as many pieces as the last position
        //from the previous game !!!!
        //Those kinds of errors will go away with the new game_format
        //

        //TODO
        //1. I am curious and I am going to count how many games are effected by that bug (which is
        //   just because of my poor judgement)
        //2. Check once again if result and position are consistent
        //3. Transfer the entire dataset to the new format
        //4. Implement functions to rescore the new dataset

        //I will analyze this again and look at more games !
        //IDEA: Printing a small window around the positions, where the error occurs !!!
        for (index, sample) in pos_iter.enumerate() {
            if index == 0 {
                test_game.set_result(sample.result);
                test_game.set_start_position(sample.position);
            } else {
                let added = test_game.add_position(sample.position);
                if added == None {
                    counter += 1;
                    if (counter % 10 == 0) {
                        println!("Counter: {}", (counter as f32) / (game_index as f32));
                    }
                    continue 'outer;
                }
            }
            //sample.position.print_position();
            //println!();
        }
    }
    println!("Number of effected games is given by {}", counter);
    /*
    let test_game_samples = test_game.get_samples();
    for (sample_new, sample_old) in zip(test_game_samples.iter(), game.iter().rev()) {
        if sample_new.result != sample_old.result || sample_new.position != sample_old.position {
            println!("Error, samples are not the same");
        }
    }
    */

    //if this works we can dump the new games to a file
    //then we can rework the game iterator
    //
    */

    // Data::convert_samples_to_games("/mnt/e/Iamhere8.samples", "/mnt/e/Iamhere8.games");
    //Data::print_samples_new_game_format("/mnt/e/Iamhere8.games")?;
    Data::rescore_games(
        "E:\\Iamhere8.games",
        "E:\\nextformattest.rescored.samples",
        &base,
    )?;

    Ok(())
}
