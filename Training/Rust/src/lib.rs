#![feature(buf_read_has_data_left)]
use pyo3::prelude::*;

use Pos::{Position, Square};
pub mod Pos;
pub mod Sample;
pub mod dataloader;
use itertools::Itertools;
use numpy::{PyArray, PyArray1, PyArrayMethods};
//Wrapper for the dataloader

use dataloader::{DataLoader, PolicyDataLoader};

#[pyclass]
struct PolicyBatchProvider {
    loader: PolicyDataLoader,
    batch_size: usize,
}

#[pymethods]
impl PolicyBatchProvider {
    #[new]
    #[pyo3(signature = (path, size, bsize, shuffle, rotate=false))]
    fn new(path: String, size: usize, bsize: usize, shuffle: bool, rotate: bool) -> Self {
        PolicyBatchProvider {
            loader: PolicyDataLoader::new(path, size, shuffle, rotate).expect("Error could not load"),
            batch_size: bsize,
        }
    }

    #[getter(num_samples)]
    fn get_samples(&self) -> PyResult<i64> {
        Ok(self.loader.num_samples as i64)
    }

    fn fill(
        &mut self,
        py: Python<'_>,
        input: Bound<'_, PyArray1<f32>>,   // B*120
        eval: Bound<'_, PyArray1<f32>>,    // B
        bucket: Bound<'_, PyArray1<i64>>,  // B
        q_score: Bound<'_, PyArray1<f32>>, // B*128, zero-initialised
        q_mask: Bound<'_, PyArray1<bool>>, // B*128, zero-initialised
    ) -> PyResult<()> {
        // raw slices are taken while holding the GIL; the actual work runs with
        // the GIL released (py.allow_threads), so the Python main thread can keep
        // launching GPU work while a loader thread fills the next batch
        let not_contiguous =
            |_| pyo3::exceptions::PyValueError::new_err("fill: arrays must be contiguous");
        let (in_array, eval_array, bucket_array, q_score_array, q_mask_array) = unsafe {
            (
                input.as_slice_mut().map_err(not_contiguous)?,
                eval.as_slice_mut().map_err(not_contiguous)?,
                bucket.as_slice_mut().map_err(not_contiguous)?,
                q_score.as_slice_mut().map_err(not_contiguous)?,
                q_mask.as_slice_mut().map_err(not_contiguous)?,
            )
        };
        let batch_size = self.batch_size;
        let loader = &mut self.loader;
        py.allow_threads(move || {
            for i in 0..batch_size {
                let ps = loader.get_next().expect("Error loading sample");
                let mut sample = ps.sample;
                if sample.position.color == -1 {
                    sample.position = sample.position.get_color_flip();
                }

                for square in sample.position.iter() {
                    match square {
                        Square::WPAWN(ix) => in_array[120 * i + ix as usize - 4] = 1.0,
                        Square::BPAWN(ix) => in_array[120 * i + ix as usize + 28] = 1.0,
                        Square::WKING(ix) => in_array[120 * i + ix as usize + 56] = 1.0,
                        Square::BKING(ix) => in_array[120 * i + ix as usize + 88] = 1.0,
                    }
                }

                eval_array[i] = sample.value.clamp(-500, 500) as f32;
                bucket_array[i] = sample.position.bucket_index();

                for c in ps.children.iter() {
                    if c.capture {
                        continue;
                    }
                    let idx = 128 * i + c.enc as usize;
                    q_score_array[idx] = Sample::policy_score(c.score);
                    q_mask_array[idx] = true;
                }
            }
        });
        Ok(())
    }

    /// Compact batch: per sample the 3 bitboards (white to move, as i64), the
    /// clamped eval and the bucket; the non-capture children as flat lists
    /// (row, encoding, policy score). The GPU expands them into the 120 inputs and
    /// the 128 scores/mask (Experimental.expand_compact). Returns the number of
    /// children written. Same samples and values as fill().
    fn fill_compact(
        &mut self,
        py: Python<'_>,
        pos: Bound<'_, PyArray1<i64>>,         // B*3: wp, bp, k
        eval: Bound<'_, PyArray1<f32>>,        // B
        bucket: Bound<'_, PyArray1<i64>>,      // B
        child_row: Bound<'_, PyArray1<i64>>,   // capacity
        child_enc: Bound<'_, PyArray1<i64>>,   // capacity
        child_score: Bound<'_, PyArray1<f32>>, // capacity
    ) -> PyResult<usize> {
        let not_contiguous =
            |_| pyo3::exceptions::PyValueError::new_err("fill_compact: arrays must be contiguous");
        let (pos_a, eval_a, bucket_a, row_a, enc_a, score_a) = unsafe {
            (
                pos.as_slice_mut().map_err(not_contiguous)?,
                eval.as_slice_mut().map_err(not_contiguous)?,
                bucket.as_slice_mut().map_err(not_contiguous)?,
                child_row.as_slice_mut().map_err(not_contiguous)?,
                child_enc.as_slice_mut().map_err(not_contiguous)?,
                child_score.as_slice_mut().map_err(not_contiguous)?,
            )
        };
        let batch_size = self.batch_size;
        let loader = &mut self.loader;
        let cap = row_a.len().min(enc_a.len()).min(score_a.len());
        let n = py.allow_threads(move || -> Result<usize, String> {
            let mut n = 0usize;
            for i in 0..batch_size {
                let ps = loader.get_next().map_err(|e| e.to_string())?;
                let mut sample = ps.sample;
                if sample.position.color == -1 {
                    sample.position = sample.position.get_color_flip();
                }
                pos_a[3 * i] = sample.position.wp as i64;
                pos_a[3 * i + 1] = sample.position.bp as i64;
                pos_a[3 * i + 2] = sample.position.k as i64;
                eval_a[i] = sample.value.clamp(-500, 500) as f32;
                bucket_a[i] = sample.position.bucket_index();
                for c in ps.children.iter() {
                    if c.capture {
                        continue;
                    }
                    if n >= cap {
                        return Err(format!("fill_compact: more than {} children in a batch", cap));
                    }
                    row_a[n] = i as i64;
                    enc_a[n] = c.enc as i64;
                    score_a[n] = Sample::policy_score(c.score);
                    n += 1;
                }
            }
            Ok(n)
        });
        n.map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Value-net batch from the same POL1 files: per sample the 3 bitboards (white
    /// to move, as i64), the root value (clamped to +-500 exactly like the old value
    /// training, BatchProvider.testing) and the bucket; the child lists are skipped.
    /// The GPU expands the bitboards into the 120 inputs (Experimental.expand_inputs).
    fn fill_value(
        &mut self,
        py: Python<'_>,
        pos: Bound<'_, PyArray1<i64>>,  // B*3: wp, bp, k
        eval: Bound<'_, PyArray1<f32>>, // B
        bucket: Bound<'_, PyArray1<i64>>, // B
    ) -> PyResult<()> {
        let not_contiguous =
            |_| pyo3::exceptions::PyValueError::new_err("fill_value: arrays must be contiguous");
        let (pos_a, eval_a, bucket_a) = unsafe {
            (
                pos.as_slice_mut().map_err(not_contiguous)?,
                eval.as_slice_mut().map_err(not_contiguous)?,
                bucket.as_slice_mut().map_err(not_contiguous)?,
            )
        };
        let batch_size = self.batch_size;
        let loader = &mut self.loader;
        let r = py.allow_threads(move || -> Result<(), String> {
            for i in 0..batch_size {
                let ps = loader.get_next().map_err(|e| e.to_string())?;
                let mut sample = ps.sample;
                if sample.position.color == -1 {
                    sample.position = sample.position.get_color_flip();
                }
                pos_a[3 * i] = sample.position.wp as i64;
                pos_a[3 * i + 1] = sample.position.bp as i64;
                pos_a[3 * i + 2] = sample.position.k as i64;
                eval_a[i] = sample.value.clamp(-500, 500) as f32;
                bucket_a[i] = sample.position.bucket_index();
            }
            Ok(())
        });
        r.map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

#[pyclass]
struct BatchProvider {
    loader: DataLoader,
    batch_size: usize,
}

//need to add a helper function which generates network input given a fen_string
//to be continued

/// Prints `num_samples` random positions of a policy file with their soft targets.
#[pyfunction]
#[pyo3(signature = (path, num_samples, temperature=50.0))]
fn print_soft_targets(path: &str, num_samples: usize, temperature: f32) -> PyResult<()> {
    dataloader::print_soft_targets(path, num_samples, temperature)?;
    Ok(())
}

#[pyfunction]
fn print_fen_string(fen_string: &str) -> PyResult<()> {
    let position = Pos::Position::try_from(fen_string)?;
    position.print_position();

    Ok(())
}

#[pyfunction]
fn input_from_fen(input: Bound<'_, PyArray1<f32>>, fen_string: &str) -> PyResult<i32> {
    let mut position =
        Position::try_from(fen_string).expect("Could not create position from fen_string");
    //need to invert to the correct color

    if position.color == -1 {
        position = position.get_color_flip();
    }

    let piece_count = position.piece_count();
    unsafe {
        let mut in_array = input.as_array_mut();
        for square in position.iter() {
            match square {
                Square::WPAWN(index) => {
                    in_array[index as usize - 4] = 1.0;
                }
                Square::BPAWN(index) => {
                    in_array[index as usize + 28] = 1.0;
                }
                Square::WKING(index) => {
                    in_array[index as usize + 28 + 28] = 1.0;
                }
                Square::BKING(index) => {
                    in_array[index as usize + 28 + 28 + 32] = 1.0;
                }
            }
        }
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
        Ok(sub_two)
    }
}

#[pymethods]
impl BatchProvider {
    #[new]
    fn new(path: String, size: usize, bsize: usize, shuffle: bool) -> Self {
        println!("{}", path);
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
        input: Bound<'_, PyArray1<f32>>,
        legal_mask: Bound<'_, PyArray1<bool>>,
        result: Bound<'_, PyArray1<f32>>,
        eval: Bound<'_, PyArray1<f32>>,
        mlh: Bound<'_, PyArray1<i64>>,
        bucket: Bound<'_, PyArray1<i64>>,
    ) -> PyResult<()> {
        unsafe {
            //make sure that on the python side
            //I pass zerod arrays (for most of them)
            let mut in_array = input.as_array_mut();
            let mut legal_array = legal_mask.as_array_mut();
            let mut res_array = result.as_array_mut();
            let mut eval_array = eval.as_array_mut();
            let mut bucket_array = bucket.as_array_mut();
            let mut mlh_array = mlh.as_array_mut();
            for i in 0..self.batch_size {
                let mut indices = Vec::with_capacity(128);
                //need to add continue for not valid samples
                let mut sample = self.loader.get_next().expect("Error loading sample");

                if sample.position.color == -1 {
                    sample.position = sample.position.get_color_flip();
                }
                sample.value = sample.value.clamp(-500, 500);

                let mut liste = Pos::MoveList::new();
                liste.get_moves(sample.position);
                for legal in liste.iter() {
                    indices.push(legal.get_move_encoding() as usize);
                }

                for index in indices {
                    legal_array[128 * i + index] = true;
                }

                for square in sample.position.iter() {
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

                mlh_array[i] = sample.mlh as i64;
                eval_array[i] = sample.value as f32;

                bucket_array[i] = sample.position.bucket_index();
            }
        }
        Ok(())
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn string_sum<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<BatchProvider>()?;
    m.add_class::<PolicyBatchProvider>()?;
    m.add_function(wrap_pyfunction!(print_soft_targets, m)?)?;
    m.add_function(wrap_pyfunction!(print_fen_string, m)?)?;
    m.add_function(wrap_pyfunction!(input_from_fen, m)?)?;
    Ok(())
}
