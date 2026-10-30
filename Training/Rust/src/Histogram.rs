use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

const RECORD_SIZE: usize = 16; // bytes per sample in your file
const MAX_PIECES: usize = 24; // 12 per side in checkers

#[derive(Default)]
pub struct Histogram {
    counts: [u64; MAX_PIECES + 1],
    total: u64,
}

impl Histogram {
    pub fn add(&mut self, black: u32, white: u32) {
        let n = (black | white).count_ones() as usize;
        self.counts[n.min(MAX_PIECES)] += 1;
        self.total += 1;
    }

    pub fn merge(&mut self, other: &Histogram) {
        for i in 0..=MAX_PIECES {
            self.counts[i] += other.counts[i];
        }
        self.total += other.total;
    }

    pub fn print(&self) {
        println!("pieces      count     share");
        for (n, &c) in self.counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let share = 100.0 * c as f64 / self.total as f64;
            let bar = "#".repeat((share / 2.0).round() as usize);
            println!("{:>6} {:>12} {:>8.3}%  {}", n, c, share, bar);
        }
        println!("total  {:>12}", self.total);
    }
}
