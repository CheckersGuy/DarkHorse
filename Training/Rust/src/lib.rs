#![feature(buf_read_has_data_left)]
use pyo3::{exceptions::PyValueError, prelude::*};
use std::borrow::BorrowMut;
use Pos::{Position, Square};
pub mod Pos;
pub mod Sample;
pub mod dataloader;
use dataloader::DataLoader;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
//Wrapper for the dataloader
#[pyclass]
struct BatchProvider {
    loader: DataLoader,
    batch_size: usize,
}

#[pymethods]
impl BatchProvider {
    #[new]
    fn new(path: String, size: usize, bsize: usize, shuffle: bool) -> Self {
        let result = BatchProvider {
            loader: DataLoader::new(path, size, shuffle).expect("Error could not load"),
            batch_size: bsize,
        };
        result
    }
    #[getter(num_samples)]
    fn get_samples(&self) -> PyResult<i32> {
        Ok(self.loader.num_samples as i32)
    }
    fn testing(
        &mut self,
        py: Python<'_>,
        input: &PyArray1<f32>,
        result: &PyArray1<f32>,
        evals: &PyArray1<i16>,
        bucket: &PyArray1<i64>,
        psqt_buckets: &PyArray1<i64>,
    ) -> PyResult<()> {
        unsafe {
            let mut in_array = input.as_array_mut();
            let mut res_array = result.as_array_mut();
            let mut bucket_array = bucket.as_array_mut();
            let mut psqt_array = psqt_buckets.as_array_mut();
            let mut eval_array = evals.as_array_mut();
            for i in 0..self.batch_size {
                //need to add continue for not valid samples
                let sample = self.loader.get_next().expect("Error loading sample");
                //below seriously needs to be checked
                /*let board_index = |index: usize| {
                    let temp = Pos::BOARD_BIT[index];
                    let row = temp / 4;
                    let col = temp % 4;
                    4 * row + 3 - col
                    //4 * row + col
                };
                */
                //above changed from 4*row+3-col
                /*let mut position = String::new();
                if let Sample::SampleType::Fen(ref fen_string) = sample.position {
                    position = fen_string.clone();
                } else {
                    println!("I suck at error handling");
                }
                */
                /*
                for square in position.iter() {
                    match square {
                        Square::WPAWN(index) => {
                            in_array[120 * i + board_index(index) - 4] = 1.0;
                        }
                        Square::BPAWN(index) => {
                            in_array[120 * i + board_index(index) + 28] = 1.0;
                        }
                        Square::WKING(index) => {
                            in_array[120 * i + board_index(index) + 28 + 28] = 1.0;
                        }
                        Square::BKING(index) => {
                            in_array[120 * i + board_index(index) + 28 + 28 + 32] = 1.0;
                        }
                    }
                }
                */
                let mut piece_count: usize = 0;
                let squares = sample.position.get_squares().unwrap();
                for square in squares {
                    match square {
                        Square::WPAWN(index) => {
                            in_array[120 * i + index - 4] = 1.0;
                        }
                        Square::BPAWN(index) => {
                            in_array[120 * i + index + 28] = 1.0;
                        }
                        Square::WKING(index) => {
                            in_array[120 * i + index + 28 + 28] = 1.0;
                        }
                        Square::BKING(index) => {
                            in_array[120 * i + index + 28 + 28 + 32] = 1.0;
                        }
                    }
                    piece_count += 1;
                }

                match sample.result {
                    Sample::Result::WIN | Sample::Result::TBWIN => res_array[i] = 1.0,
                    Sample::Result::LOSS | Sample::Result::TBLOSS => res_array[i] = 0.0,
                    Sample::Result::DRAW | Sample::Result::TBDRAW => res_array[i] = 0.5,
                    _ => (), //need to add error handling just go to the nex sample in that case
                }

                eval_array[i] = sample.eval;

                let psqt_index = (piece_count - 1) / 4;
                psqt_array[i] = psqt_index as i64;
                let sub_two;
                match piece_count {
                    24 | 23 | 22 | 21 | 20 | 19 => sub_two = 0,
                    18 | 17 | 16 => sub_two = 1,
                    15 | 14 | 13 => sub_two = 2,
                    12 | 11 => sub_two = 3,
                    10 => sub_two = 4,
                    9 => sub_two = 5,
                    8 => sub_two = 6,
                    7 => sub_two = 7,
                    6 => sub_two = 8,
                    5 => sub_two = 9,
                    4 => sub_two = 10,
                    3 | 2 | 1 | 0 => sub_two = 11,
                    _ => sub_two = 0,
                }
                bucket_array[i] = sub_two;
                //testing
            }
        }
        Ok(())
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn string_sum(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<BatchProvider>()?;
    Ok(())
}
