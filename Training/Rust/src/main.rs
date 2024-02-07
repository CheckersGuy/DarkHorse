#![feature(buf_read_has_data_left)]
#![feature(iter_next_chunk)]
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
use Sample::SampleType;
use TableBase::Base;
fn main() -> anyhow::Result<()> {
    /* let mut dataloader = dataloader::DataLoader::new(
            String::from("E:/newtry11rescoredmlh.samples"),
            1000000,
            false,
        )?;

        for _ in 0..3000 {
            let sample = dataloader.get_next()?;
            println!("{:?}", sample);
            if let Sample::SampleType::Fen(ref position) = sample.position {
                let pos = Pos::Position::try_from(position.as_str())?;
                pos.print_position();
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
    //let fen_string = "B:W30,29:B4,24";
    let base = Base::new_dtw("E:\\kr_english_wld", "E:\\kr_english_dtw", 2000, 10).unwrap();
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
    /*
        Data::rescore_games(
            "E:/newtry11rescored.samples",
            "E:/newtry11rescoredmlh.samples",
            &base,
        )
        .unwrap();
    */
    //Data::create_unique_fens("newopen2.pos", "newopen3.pos").unwrap();
    // Data::create_book("../Positions/drawbook.book", "newopen3.pos", 14)?;

    /*Data::dump_mlh_samples(
            "E:/newtry11rescoredmlh.samples",
            "E:/newtry11rescoredmlhwinning.samples",
        )?;
    */
    let mut reader = BufReader::new(File::open("E:/newtry11rescoredmlh.samples")?);
    for game in reader.iter_games().take(1) {
        for window in game.windows(2) {
            let fen_next = match window[0].position {
                SampleType::Fen(ref fen_string) => fen_string.clone(),
                _ => String::new(),
            };
            let fen_previous = match window[1].position {
                SampleType::Fen(ref fen_string) => fen_string.clone(),
                _ => String::new(),
            };
            let move_encoding = base
                .get_move_encoding(fen_previous.as_str(), fen_next.as_str())
                .unwrap();
            println!("{move_encoding}");

            if move_encoding > 0 {
                base.print_fen(fen_previous.as_str());
                println!("{move_encoding}");
            }
        }

        //base.print_fen(fen_string.as_str()).unwrap();
    }

    Ok(())
}
