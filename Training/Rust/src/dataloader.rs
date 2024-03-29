use crate::Sample;
use crate::Sample::SampleIteratorTrait;

use byteorder::{LittleEndian, ReadBytesExt};
use rand::prelude::*;
use rayon::prelude::*;
use rip_shuffle::RipShuffleParallel;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::time::Instant;
use Sample::SampleType;
#[derive(Debug)]

pub struct DataLoader {
    reader: std::io::BufReader<std::fs::File>,
    pub path: String,
    shuff_buf: Vec<Sample::Sample>,
    shuffle: bool,
    pub num_samples: Option<u64>,
    capa: usize,
    rng: StdRng,
}

impl DataLoader {
    pub fn new(path: String, capacity: usize, shuffle: bool) -> std::io::Result<DataLoader> {
        let mut data_loader = DataLoader {
            reader: BufReader::with_capacity(1000000, File::open(path.clone())?),
            path: path.clone(),
            shuff_buf: Vec::new(),
            num_samples: None,
            shuffle,
            capa: capacity,
            rng: StdRng::from_rng(thread_rng()).unwrap(),
        };
        {
            let mut reader = BufReader::new(File::open(path)?);
            data_loader.num_samples = Some(reader.iter_samples().count() as u64);
        };

        if let Some(num_samples) = data_loader.num_samples {
            data_loader.capa = std::cmp::min(data_loader.capa, num_samples as usize);
            data_loader.shuff_buf = Vec::with_capacity(data_loader.capa);
            println!("Got {} available samples", num_samples);
        }
        Ok(data_loader)
    }

    pub fn read(&mut self) -> std::io::Result<Sample::Sample> {
        let has_data_left = self.reader.has_data_left()?;
        if !has_data_left {
            println!("Reached the end of the file and buffer is empty");
            self.reader.rewind()?;
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
            self.shuff_buf.par_iter_mut().for_each(|sample| {
                if let SampleType::Fen(ref fen_string) = sample.position {
                    sample.position = SampleType::Squares(sample.position.get_squares().unwrap());
                }
            });

            let elapsed = now.elapsed().as_millis();
            println!("Elapsed time {}", elapsed);
            println!("Transformation time {}", transform.elapsed().as_millis());
        }

        let sample = self.shuff_buf.pop().unwrap();
        Ok(sample)
    }

    pub fn dump_pos_to_file(&mut self, output: String) -> std::io::Result<()> {
        self.shuffle = false;
        let mut writer = File::create(output)?;
        while let Ok(sample) = self.get_next() {
            match sample.position {
                SampleType::Fen(fen_string) => {
                    writer.write_all((fen_string + "\n").as_bytes())?;
                }
                SampleType::Pos(_) => (),
                _ => (),
            }
        }

        Ok(())
    }
}
