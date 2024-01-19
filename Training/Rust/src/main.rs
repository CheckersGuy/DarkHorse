#![feature(buf_read_has_data_left)]
pub mod Data;
pub mod Pos;
pub mod Sample;
pub mod TableBase;
pub mod dataloader;
use std::fs::File;
use std::io::BufReader;
use Data::count_unique_samples;
use Data::Generator;
use Sample::SampleIteratorTrait;
use TableBase::Base;
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
    /* Data::merge_samples(
        vec![
            "../TrainData/newopen14.samples",
            "../TrainData/merged.samples",
        ],
        "../TrainData/merged2.samples",
    )?;
    */
    //let distribution = Data::material_distrib("../TrainData/testing2.samples")?;
    //println!("{:?}", distribution);
    /*
        let mut generator = Generator::new(
            String::from("../Positions/newopen4.pos"),
            String::from("../TrainData/newtry.samples"),
            14,
            1000000,
        );
        generator.time = 10;
        //generator.prev_file = Some("../TrainData/usetablebase4.samples");
        generator.generate_games()?;
    */
    let base = Base::new("E:\\kr_english_wld", 100, 10).unwrap();
    /*
        let result = base.probe("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        base.print_fen("W:W8,6,K13:BK4,7,5,11,9").unwrap();
        println!("{:?}", result);
    */

    Data::rescore_games(
        "../TrainData/newtry.samples",
        "../TrainData/newtryrescored.samples",
        &base,
    )
    .unwrap();
    /*
        let squares = SampleType::Fen(String::from("B:WK19:BK15,K18"))
            .get_squares()
            .unwrap();

        for square in squares {
            println!("{:?}", square);
        }
    */

    /* let pos = Pos::Position::try_from("B:W20,21,22,24,25,26,27,30,31,32:B2,3,4,7,8,10,11,12,14,15")
            .unwrap();
        pos.print_position();
    */
    //Data::create_unique_fens("newopen2.pos", "newopen3.pos").unwrap();
    // Data::create_book("../Positions/drawbook.book", "newopen3.pos", 14)?;
    //

    Ok(())
}
