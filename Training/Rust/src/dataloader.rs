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

/// Policy samples in shuffle buffers of `capacity` samples, read sequentially from
/// the file (wrapping around at the end). A background thread reads and shuffles
/// the NEXT buffer while the current one is consumed, so get_next() no longer
/// stalls for seconds at every refill. Sampling is unchanged: sequential read,
/// buffer shuffled (if `shuffle`), samples taken from the end of the buffer.
pub struct PolicyDataLoader {
    shuff_buf: Vec<PolicySample>,
    pub num_samples: u64,
    rx: std::sync::mpsc::Receiver<std::io::Result<Vec<PolicySample>>>,
}


/// Every INDEX_STRIDE-th record's byte offset is remembered during the first pass
/// (records have variable length), so later passes can start at a random record.
const INDEX_STRIDE: u64 = 4096;

impl PolicyDataLoader {
    /// `rotate`: after every full pass over the file (num_samples records) the next
    /// pass starts at a random record (one of every INDEX_STRIDE) and wraps around,
    /// so every pass still reads every record exactly once, but the shuffle buffers
    /// (blocks of `capacity` consecutive records) differ from pass to pass.
    pub fn new(path: String, capacity: usize, shuffle: bool, rotate: bool) -> std::io::Result<Self> {
        let file_len = std::fs::metadata(&path)?.len();
        let mut reader = BufReader::with_capacity(1_000_000, File::open(&path)?);
        let num_samples = read_policy_header(&mut reader)?;
        let capa = std::cmp::min(num_samples as usize, capacity);
        // rendezvous channel: at most one finished buffer waits besides the one in use
        let (tx, rx) = std::sync::mpsc::sync_channel(0);
        std::thread::spawn(move || {
            let mut rng = StdRng::from_rng(thread_rng()).unwrap();
            let mut pos = POLICY_HEADER_BYTES; // byte offset of the next record
            let mut ordinal: u64 = 0; // file index of the next record
            let mut in_pass: u64 = 0; // records read in the current pass
            let mut index: Vec<(u64, u64)> = Vec::new(); // (ordinal, byte offset)
            let mut index_done = false;
            loop {
                let mut buf = Vec::with_capacity(capa);
                for _ in 0..capa {
                    if rotate && in_pass == num_samples {
                        index_done = true;
                        in_pass = 0;
                        if !index.is_empty() {
                            let (o, p) = index[rng.gen_range(0..index.len())];
                            if let Err(e) = reader.seek(SeekFrom::Start(p)) {
                                let _ = tx.send(Err(e));
                                return;
                            }
                            ordinal = o;
                            pos = p;
                        }
                    }
                    if pos >= file_len {
                        // wrap to the first record (NOT rewind(): that is the header)
                        if let Err(e) = reader.seek(SeekFrom::Start(POLICY_HEADER_BYTES)) {
                            let _ = tx.send(Err(e));
                            return;
                        }
                        pos = POLICY_HEADER_BYTES;
                        ordinal = 0;
                        if !rotate {
                            index_done = true;
                        }
                    }
                    if rotate && !index_done && ordinal % INDEX_STRIDE == 0 {
                        index.push((ordinal, pos));
                    }
                    let mut ps = PolicySample::new(Sample::Sample::default());
                    match ps.read_into(&mut reader) {
                        Ok(()) => {
                            pos += 19 + 4 * ps.children.len() as u64;
                            ordinal += 1;
                            in_pass += 1;
                            buf.push(ps)
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            return;
                        }
                    }
                }
                if shuffle {
                    buf.par_shuffle(&mut rng);
                }
                if tx.send(Ok(buf)).is_err() {
                    return; // loader dropped
                }
            }
        });
        Ok(PolicyDataLoader {
            shuff_buf: Vec::new(),
            num_samples,
            rx,
        })
    }

    pub fn get_next(&mut self) -> std::io::Result<PolicySample> {
        if self.shuff_buf.is_empty() {
            self.shuff_buf = self.rx.recv().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::Other, "policy loader thread ended")
            })??;
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
