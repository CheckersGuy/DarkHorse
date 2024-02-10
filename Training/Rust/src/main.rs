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
    /*let mut dataloader =
            dataloader::DataLoader::new(String::from("/mnt/e/mlhshuffled.samples"), 1000000, false)?;

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

    /*Data::create_policy_data(
        "E:/newtry11rescoredmlhshuffled.samples",
        "E:/policyshuffled.samples",
        &base,
    )
    .unwrap();
    */
    /*Data::dump_mlh_samples(
            "/mnt/e/newtry11rescoredmlhshuffled.samples",
            "/mnt/e/mlhshuffled2.samples",
        )?;

    */

    Data::create_mlh_data(
        "/mnt/e/newtry11rescored.samples",
        "/mnt/e/mlh3.samples",
        &base,
    )?;

    Ok(())
}
