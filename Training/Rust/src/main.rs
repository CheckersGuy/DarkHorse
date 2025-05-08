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

    //
    /*let mut generator = Generator::new(
        String::from("../Positions/ultrabook2.pos"),
        String::from("/mnt/e/Iamhere7.samples"),
        14,
        500000000,
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
    /*
    let test = Position::try_from("B:WK2,29:BK3,K12").expect("Could not parse fen");
    test.print_position();
    println!();
    let result = base
        .probe("B:WK2,29:BK3,K12")
        .expect("Could not call function");

    println!("{:?}", result);

    let result2 = base
        .probe_with_position(test)
        .expect("Could not call function");

    println!("Result using probing with position: {:?}", result2);
    println!("Color: {:?}", test.color);
    println!("{:?}", test.wp);
    println!("{:?}", test.bp);
    println!("{:?}", test.k);

    let test_position =
        Position::try_from("W:W10,28,29:BK8,K7,K24,K25").expect("Could not parse fen");

    for square in test_position.iter() {
        println!("{:?}", square);
    }
    */

    /*
        let result = base.probe("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        base.print_fen("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        println!("{:?}", result);
    */

    /*Data::dump_mlh_samples(
            "/mnt/e/newtry11rescoredmlhshuffled.samples",
            "/mnt/e/mlhshuffled2.samples",
        )?;

    */

    //Data::create_mlh_data("E:/newtry11rescored.samples", "E:/mlh3.samples", &base)?;
    /* Data::shuffle_data_external::<16>(
         "/mnt/e/final1mstestrescored.samples",
         "/mnt/e/final1msshuffled.samples",
     )?;
    */

    Data::merge_rescored_data(
        vec![
            "/mnt/e/Iamherenextrescored.samples",
            "/mnt/e/Iamhere7rescored.samples",
        ],
        "/mnt/e/Iamherenext2rescored.samples",
    )?;

    Data::shuffle_data_external::<16>(
        "/mnt/e/Iamherenext2rescored.samples",
        "/mnt/e/Iamherenext2shuffled.samples",
    )?;
    /* Data::rescore_games(
         "E:\\testnodes4.samples",
         "E:\\testnodes4rescoredcheck.samples",
         &base,
     )?;
    */

    Ok(())
}
