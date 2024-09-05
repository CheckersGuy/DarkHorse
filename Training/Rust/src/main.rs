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
    /*let mut reader = BufReader::new(File::open("E:\\newformatrescored.samles")?);

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
    let mut generator = Generator::new(
        String::from("../Positions/ultrabook2.pos"),
        String::from("/mnt/e/testnode6.samples"),
        14,
        150000000,
    );

    generator.time = 100000;
    generator.max_nodes = 50000000;
    generator.depth = 7;

    generator.generate_games()?;

    //generator.prev_file = Some("/mnt/e/finalrescored/paritysuperiorityshuffled.samples");

    /*Data::create_subset(
            "/mnt/e/policyultimateshuffled.samples",
            "/mnt/e/smallpolicy.samples",
            2000000,
        )
        .unwrap();
    */
    //Data::create_book("../Positions/drawbook.book", "testbook.pos", 6)?;

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
    /*Data::shuffle_data_external::<32>(
        "/mnt/e/testnodes4rescored.samles",
        "/mnt/e/testnodes4shuffle3d.samples",
    )?;
    */

    /*Data::merge_rescored_data(
        vec![
            "/mnt/e/finalms500batch2rescored.samples",
            "/mnt/e/finalms500rescored.samples",
        ],
        "/mnt/e/final1rescored.samples",
    );

    */

    /*Data::rescore_games(
        "E:\\testnodes4.samples",
        "E:\\testnodes4rescored.samles",
        &base,
    )?;
    */

    Ok(())
}
