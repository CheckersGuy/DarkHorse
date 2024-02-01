#![feature(buf_read_has_data_left)]
pub mod Data;
pub mod Pos;
pub mod Sample;
pub mod TableBase;
pub mod dataloader;
use anyhow::Context;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use Data::count_unique_samples;
use Data::Generator;
use Pos::Square;
use Sample::SampleIteratorTrait;
use TableBase::Base;
fn main() -> anyhow::Result<()> {
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
    /* Data::merge_samples(
        vec![
            "../TrainData/newopen14.samples",
            "../TrainData/merged.samples",
        ],
        "../TrainData/merged2.samples",
    )?;
    */

    /*
        let mut generator = Generator::new(
            String::from("../Positions/newopen4.pos"),
            String::from("/mnt/e/newtry10.samples"),
            14,
            40000000,
        );
        generator.time = 10;
        generator.prev_file = Some("/mnt/e/newtry7rescored.samples");
        generator.generate_games()?;
    */
    /*
        println!(
            "{}",
            Sample::SampleType::invert_fen_string("B:W22,K26,K27:B6,7,K15").unwrap()
        );
    */
    //let base = Base::new("E:\\kr_english_wld", 100, 10).unwrap();
    /*
        let result = base.probe("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        base.print_fen("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        println!("{:?}", result);
    */

    // Data::rescore_games("E:/newtry9.samples", "E:/newtry9rescored.samples", &base).unwrap();

    //Data::create_unique_fens("newopen2.pos", "newopen3.pos").unwrap();
    // Data::create_book("../Positions/drawbook.book", "newopen3.pos", 14)?;
    //

    Ok(())
}
