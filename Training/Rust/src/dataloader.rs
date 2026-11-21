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

use crate::Sample::{read_policy_header, PolicySample, POLICY_HEADER_BYTES};
use std::io::SeekFrom;

pub struct PolicyDataLoader {
    reader: BufReader<File>,
    shuff_buf: Vec<PolicySample>,
    shuffle: bool,
    pub num_samples: u64,
    capa: usize,
    rng: StdRng,
}

impl PolicyDataLoader {
    pub fn new(path: String, capacity: usize, shuffle: bool) -> std::io::Result<Self> {
        let mut reader = BufReader::with_capacity(1_000_000, File::open(&path)?);
        let num_samples = read_policy_header(&mut reader)?;
        Ok(PolicyDataLoader {
            reader,
            shuff_buf: Vec::new(),
            shuffle,
            num_samples,
            capa: std::cmp::min(num_samples as usize, capacity),
            rng: StdRng::from_rng(thread_rng()).unwrap(),
        })
    }

    fn read(&mut self) -> std::io::Result<PolicySample> {
        if !self.reader.has_data_left()? {
            // NOT rewind(): that would put us back inside the header
            self.reader.seek(SeekFrom::Start(POLICY_HEADER_BYTES))?;
        }
        let mut ps = PolicySample::new(Sample::Sample::default());
        ps.read_into(&mut self.reader)?;
        Ok(ps)
    }

    pub fn get_next(&mut self) -> std::io::Result<PolicySample> {
        if self.shuff_buf.is_empty() {
            for _ in 0..self.capa {
                let s = self.read()?;
                self.shuff_buf.push(s);
            }
            if self.shuffle {
                self.shuff_buf.par_shuffle(&mut self.rng);
            }
        }
        Ok(self.shuff_buf.pop().unwrap())
    }
}

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
        let packed_size_on_disk: u64 = 18;
        data_loader.num_samples = Some(file_length / packed_size_on_disk);
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

/// Prints `num_samples` random positions from a policy file (as written by
/// `rescoring_children`, with header) together with their soft targets.
///
/// Soft targets are computed exactly like in training (Experimental.py):
/// softmax(policy_score(score) / temperature) over the non-capture children.
/// Rows with fewer than 2 such children are skipped, since training drops them too.
/// Positions are picked by reservoir sampling over the whole file.
pub fn print_soft_targets(path: &str, num_samples: usize, temperature: f32) -> std::io::Result<()> {
    use crate::Pos::MoveList;
    use std::collections::HashMap;

    let mut reader = BufReader::with_capacity(1_000_000, File::open(path)?);
    let total = read_policy_header(&mut reader)?;
    let mut rng = thread_rng();
    let mut reservoir: Vec<PolicySample> = Vec::with_capacity(num_samples);
    let mut seen: u64 = 0;
    loop {
        let mut ps = PolicySample::new(Sample::Sample::default());
        match ps.read_into(&mut reader) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        if ps.children.iter().filter(|c| !c.capture).count() < 2 {
            continue;
        }
        seen += 1;
        if reservoir.len() < num_samples {
            reservoir.push(ps);
        } else {
            let j = rng.gen_range(0..seen);
            if (j as usize) < num_samples {
                reservoir[j as usize] = ps;
            }
        }
    }
    println!(
        "{} samples in file, {} with >= 2 moves, showing {} (temperature {})",
        total,
        seen,
        reservoir.len(),
        temperature
    );

    for (n, ps) in reservoir.iter().enumerate() {
        let mut pos = ps.sample.position;
        // same as PolicyBatchProvider::fill: children are encoded from white's view
        if pos.color == -1 {
            pos = pos.get_color_flip();
        }

        // move-index -> "from-to", using the same square numbers as the FEN
        let mut list = MoveList::new();
        list.get_moves(pos);
        let mut notation: HashMap<i32, String> = HashMap::new();
        for m in list.iter() {
            notation.insert(
                m.get_move_encoding(),
                format!("{}-{}", m.get_from_index() + 1, m.get_to_index() + 1),
            );
        }

        let children: Vec<_> = ps.children.iter().filter(|c| !c.capture).collect();
        let max = children
            .iter()
            .map(|c| Sample::policy_score(c.score) / temperature)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = children
            .iter()
            .map(|c| (Sample::policy_score(c.score) / temperature - max).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let mut rows: Vec<(u8, f32, i16)> = children
            .iter()
            .zip(exps.iter())
            .map(|(c, e)| (c.enc, e / sum, c.score))
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!();
        println!("===== Sample {}/{}  (root value {}) =====", n + 1, reservoir.len(), ps.sample.value);
        pos.print_position();
        println!();
        println!("{:>5}  {:<7}  {}", "index", "move", "probability (value)");
        for (enc, p, score) in rows {
            let mv = notation.get(&(enc as i32)).map(String::as_str).unwrap_or("??");
            println!("{:>5}  {:<7}  {:.4} ({})", enc, mv, p, score);
        }
        println!();
        println!("FEN: {}", pos.get_fen_string());
    }
    Ok(())
}
