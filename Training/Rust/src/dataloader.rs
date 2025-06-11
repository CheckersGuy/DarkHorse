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
        let file = File::open(path.clone())?;
        let file_length = file.metadata().unwrap().len();
        let mut data_loader = DataLoader {
            reader: BufReader::with_capacity(1000000, file),
            path: path.clone(),
            shuff_buf: Vec::new(),
            num_samples: Some(100000000),
            shuffle,
            capa: capacity,
            rng: StdRng::from_rng(thread_rng()).unwrap(),
        };

        data_loader.num_samples =
            Some(file_length / (std::mem::size_of::<Sample::Sample>() as u64));
        data_loader.capa = std::cmp::min(
            data_loader.num_samples.unwrap_or(0) as usize,
            data_loader.capa,
        );
        return Ok(data_loader);
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
        }

        let sample = self.shuff_buf.pop().unwrap();
        Ok(sample)
    }
}
