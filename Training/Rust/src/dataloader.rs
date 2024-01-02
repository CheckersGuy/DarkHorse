
use crate::Sample;

use byteorder::{LittleEndian, ReadBytesExt};
use rand::prelude::*;
use rayon::prelude::*;
use rip_shuffle::RipShuffleParallel;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::time::{Instant};
use Sample::SampleType;
#[derive(Debug)]

pub struct SampleData {
    reader: std::io::BufReader<std::fs::File>,
    num_samples: u64,
}

pub struct DataLoader {
    reader: std::io::BufReader<std::fs::File>,
    pub path: String,
    shuff_buf: Vec<Sample::Sample>,
    shuffle: bool,
    pub num_samples: u64,
    capa: usize,
    rng: StdRng,
}

impl DataLoader {
    pub fn new(path: String, capacity: usize, shuffle: bool) -> std::io::Result<DataLoader> {
        let mut data_loader = DataLoader {
            reader: BufReader::with_capacity(10000000, File::open(path.clone())?),
            path: path.clone(),
            shuff_buf: Vec::with_capacity(capacity),
            num_samples: 0,
            shuffle,
            capa: capacity,
            rng: StdRng::from_rng(thread_rng()).unwrap(),
        };
        data_loader.num_samples = data_loader.reader.read_u64::<LittleEndian>()?;
        data_loader.capa = std::cmp::min(data_loader.capa, data_loader.num_samples as usize);
        println!("Got {} available samples", data_loader.num_samples);
        Ok(data_loader)
    }

    pub fn read(&mut self) -> std::io::Result<Sample::Sample> {
        let has_data_left = self.reader.has_data_left()?;
        if !has_data_left {
            println!("Reached the end of the file and buffer is empty");
            self.reader.rewind()?;
            self.reader.read_u64::<LittleEndian>()?;
        }

        let mut sample = Sample::Sample::default();
        sample.read_into(&mut self.reader)?;
        Ok(sample)
    }

    pub fn get_next(&mut self) -> std::io::Result<Sample::Sample> {
        if self.shuff_buf.is_empty() {
            let now = Instant::now();
            for _ in 0..self.capa {
                let result = self.read()?;
                self.shuff_buf.push(result);
            }
            if self.shuffle {
                let shuff_time = Instant::now();
                self.shuff_buf.par_shuffle(&mut self.rng);
                println!("Shuffled the buffer");
                println!("ShuffleTime {}", shuff_time.elapsed().as_millis());
            }
            //Need to convert all the samples
            let transform = Instant::now();
            /* self.shuff_buf.par_iter_mut().for_each(|sample| {
                            if let SampleType::Fen(ref fen_string) = sample.position {
                                sample.position = SampleType::Pos(Position::try_from(&fen_string[..]).unwrap());
                            }
                        });
            */
            let elapsed = now.elapsed().as_millis();
            println!("Elapsed time {}", elapsed);
            println!("Transformation time {}", transform.elapsed().as_millis());
        }

        let sample = self.shuff_buf.pop().unwrap();
        Ok(sample)
    }

    pub fn dump_pos_to_file(&mut self, output: String) -> std::io::Result<()> {
        self.shuffle = false;
        println!("We have {} positions", self.num_samples);
        let mut writer = File::create(output)?;
        while let Ok(sample) = self.get_next() {
            match sample.position {
                SampleType::Fen(fen_string) => {
                    writer.write_all((fen_string + "\n").as_bytes())?;
                }
                SampleType::Pos(_) => (),
                SampleType::None => (),
            }
        }

        Ok(())
    }
}
