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
use std::path::Path;
use std::usize;
use Data::count_unique_samples;
use Data::Generator;
use Pos::Square;
use Pos::*;
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
    //let base = Base::new("E:\\kr_english_wld", 2000, 10).unwrap();

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
    Data::shuffle_data_external::<16>(
        "E:\\Iamhere9rescored.samples",
        "E:\\Iamhere9shuffled.samples",
    )?;

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
    Data::create_policy_data("E:\\Iamhere8.samples", "E:\\Iamhere8policy.samples", &base);

    Ok(())
}
