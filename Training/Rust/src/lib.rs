#![feature(buf_read_has_data_left)]
use pyo3::prelude::*;

use Pos::Square;
pub mod Pos;
pub mod Sample;
pub mod dataloader;
use dataloader::DataLoader;
use numpy::PyArray1;
//Wrapper for the dataloader
#[pyclass]
struct BatchProvider {
    loader: DataLoader,
    batch_size: usize,
}

//need to add a helper function which generates network input given a fen_string
//to be continued

#[pyfunction]
fn print_fen_string(fen_string: &str) -> PyResult<()> {
    let position = Pos::Position::try_from(fen_string)?;
    position.print_position();

    Ok(())
}

#[pyfunction]
fn input_from_fen(input: &PyArray1<f32>, fen_string: &str) -> PyResult<i32> {
    let mut fen = Sample::SampleType::Fen(String::from(fen_string));

    //need to invert to the correct color
    let get_mover = |fen: &str| -> i32 {
        match fen.chars().next() {
            Some('W') => 1,
            Some('B') => -1,
            _ => 0,
        }
    };
    if get_mover(fen_string) == -1 {
        fen = Sample::SampleType::Fen(String::from(
            Sample::SampleType::invert_fen_string(fen_string).unwrap(),
        ));
    }
    let squares = fen.get_squares()?;
    let piece_count = squares.len();
    let mut has_kings = 0;
    unsafe {
        let mut in_array = input.as_array_mut();
        for square in squares {
            match square {
                Square::WPAWN(index) => {
                    in_array[index as usize - 4] = 1.0;
                }
                Square::BPAWN(index) => {
                    in_array[index as usize + 28] = 1.0;
                }
                Square::WKING(index) => {
                    has_kings = 1;
                    in_array[index as usize + 28 + 28] = 1.0;
                }
                Square::BKING(index) => {
                    has_kings = 1;
                    in_array[index as usize + 28 + 28 + 32] = 1.0;
                }
            }
        }
        let sub_two;
        match piece_count {
            24 | 23 => sub_two = 0,
            22 => sub_two = 1,
            21 => sub_two = 2,
            20 => sub_two = 3,
            19 => sub_two = 4,
            18 => sub_two = 5,
            17 => sub_two = 6,
            16 => sub_two = 7,
            15 => sub_two = 8,
            14 => sub_two = 9,
            13 => sub_two = 10,
            12 => sub_two = 11 + has_kings,
            11 => sub_two = 13 + has_kings,
            10 => sub_two = 15 + has_kings,
            9 => sub_two = 17 + has_kings,
            8 => sub_two = 19 + has_kings,
            7 => sub_two = 21 + has_kings,
            6 => sub_two = 23 + has_kings,
            5 | 4 | 3 | 2 | 1 => sub_two = 25 + has_kings,
            _ => sub_two = 0,
        }

        Ok(sub_two)
    }
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
        Ok(self.loader.num_samples.unwrap() as i32)
    }
    fn testing(
        &mut self,
        _py: Python<'_>,
        input: &PyArray1<f32>,
        result: &PyArray1<f32>,
        mlh: &PyArray1<i16>,
        bucket: &PyArray1<i64>,
        psqt_buckets: &PyArray1<i64>,
    ) -> PyResult<()> {
        unsafe {
            let mut in_array = input.as_array_mut();
            let mut res_array = result.as_array_mut();
            let mut bucket_array = bucket.as_array_mut();
            let mut psqt_array = psqt_buckets.as_array_mut();
            let mut mlh_array = mlh.as_array_mut();
            for i in 0..self.batch_size {
                //need to add continue for not valid samples
                let mut sample = self.loader.get_next().expect("Error loading sample");

                let squares = match sample.position {
                    Sample::SampleType::Squares(our_squares) => our_squares,
                    Sample::SampleType::Fen(_) => sample.position.get_squares().unwrap(),
                    _ => Vec::new(),
                };
                let piece_count = squares.len();

                for square in squares {
                    match square {
                        Square::WPAWN(index) => {
                            in_array[120 * i + index as usize - 4] = 1.0;
                        }
                        Square::BPAWN(index) => {
                            in_array[120 * i + index as usize + 28] = 1.0;
                        }
                        Square::WKING(index) => {
                            in_array[120 * i + index as usize + 28 + 28] = 1.0;
                        }
                        Square::BKING(index) => {
                            in_array[120 * i + index as usize + 28 + 28 + 32] = 1.0;
                        }
                    }
                }

                match sample.result {
                    Sample::Result::WIN | Sample::Result::TBWIN => res_array[i] = 1.0,
                    Sample::Result::LOSS | Sample::Result::TBLOSS => res_array[i] = 0.0,
                    Sample::Result::DRAW | Sample::Result::TBDRAW => res_array[i] = 0.5,
                    _ => (), //need to add error handling just go to the nex sample in that case
                }

                mlh_array[i] = sample.mlh;

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
    m.add_function(wrap_pyfunction!(print_fen_string, m)?)?;
    m.add_function(wrap_pyfunction!(input_from_fen, m)?)?;
    Ok(())
}
