use crate::Pos::MoveList;
use crate::Pos::Position;
use crate::Sample;
use libc;
use libloading;
use std::ffi::CString;
use std::os::raw::c_char;
pub struct Base {
    cache_size: i32,
    num_pieces: i32,
    library: libloading::Library,
}

fn foo(fen_string: &str) -> Result<u32, Box<dyn std::error::Error>> {
    unsafe {
        let lib = libloading::Library::new("libRustDll.dll")?;
        let func: libloading::Symbol<unsafe extern "C" fn(*const libc::c_char) -> u32> =
            lib.get(b"print_fen")?;

        let c_to_print = CString::new(fen_string).expect("CString failed");
        Ok(func(c_to_print.as_ptr()))
    }
}

impl Base {
    pub fn new(
        path: &str,
        cache_size: i32,
        num_pieces: i32,
    ) -> Result<Base, Box<dyn std::error::Error>> {
        unsafe {
            let base = Base {
                cache_size,
                num_pieces,
                library: libloading::Library::new("libRustDll.dll")?,
            };
            let func: libloading::Symbol<
                unsafe extern "C" fn(*const libc::c_char, libc::c_int, libc::c_int),
            > = base.library.get(b"load")?;
            let c_to_print = CString::new(path).expect("CString failed");
            func(c_to_print.as_ptr(), base.cache_size, base.num_pieces);
            Ok(base)
        }
    }

    pub fn new_dtw(
        wdl_path: &str,
        dtw_path: &str,
        cache_size: i32,
        num_pieces: i32,
    ) -> Result<Base, Box<dyn std::error::Error>> {
        unsafe {
            let base = Base {
                cache_size,
                num_pieces,
                library: libloading::Library::new("libRustDll.dll")?,
            };
            let func_dtw: libloading::Symbol<
                unsafe extern "C" fn(*const libc::c_char, libc::c_int, libc::c_int),
            > = base.library.get(b"load_dtw")?;

            let func_wdl: libloading::Symbol<
                unsafe extern "C" fn(*const libc::c_char, libc::c_int, libc::c_int),
            > = base.library.get(b"load")?;

            let wdl_to_print = CString::new(wdl_path).expect("CString failed");
            let dtw_to_print = CString::new(dtw_path).expect("CString failed");
            func_wdl(wdl_to_print.as_ptr(), base.cache_size, base.num_pieces);
            func_dtw(dtw_to_print.as_ptr(), base.cache_size, base.num_pieces);
            Ok(base)
        }
    }

    pub fn probe(&self, fen_string: &str) -> Result<Sample::Result, Box<dyn std::error::Error>> {
        unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(*const libc::c_char) -> i32> =
                self.library.get(b"probe")?;
            let c_to_print = CString::new(fen_string).expect("CString failed");
            let tb_result = func(c_to_print.as_ptr());
            Ok(match tb_result {
                0 => Sample::Result::TBWIN,
                1 => Sample::Result::TBLOSS,
                2 => Sample::Result::TBDRAW,
                3 => Sample::Result::UNKNOWN,
                _ => Sample::Result::UNKNOWN,
            })
        }
    }
    pub fn probe_with_position(
        &self,
        position: Position,
    ) -> Result<Sample::Result, Box<dyn std::error::Error>> {
        unsafe {
            let func: libloading::Symbol<
                unsafe extern "C" fn(libc::c_uint, libc::c_uint, libc::c_uint, libc::c_int) -> i32,
            > = self.library.get(b"probe_with_position")?;

            let tb_result = func(position.wp, position.bp, position.k, position.color as i32);
            Ok(match tb_result {
                0 => Sample::Result::TBWIN,
                1 => Sample::Result::TBLOSS,
                2 => Sample::Result::TBDRAW,
                3 => Sample::Result::UNKNOWN,
                _ => Sample::Result::UNKNOWN,
            })
        }
    }

    pub fn probe_dtw(&self, fen_string: &str) -> Result<Option<i32>, Box<dyn std::error::Error>> {
        unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(*const libc::c_char) -> i32> =
                self.library.get(b"probe_dtw")?;
            let c_to_print = CString::new(fen_string).expect("CString failed");
            let tb_result = func(c_to_print.as_ptr());
            if tb_result > 0 {
                return Ok(Some(tb_result));
            }
            Ok(None)
        }
    }

    pub fn probe_dtw_with_position(
        &self,
        position: Position,
    ) -> Result<Option<i32>, Box<dyn std::error::Error>> {
        unsafe {
            let func: libloading::Symbol<
                unsafe extern "C" fn(libc::c_uint, libc::c_uint, libc::c_uint, libc::c_int) -> i32,
            > = self.library.get(b"probe_dtw_with_position")?;

            let tb_result = func(position.wp, position.bp, position.k, position.color as i32);
            if tb_result > 0 {
                return Ok(Some(tb_result));
            }
            Ok(None)
        }
    }

    pub fn probe_dtw_recursive(
        &self,
        position: Position,
        orig_mover: i8,
        ply: i32,
    ) -> Result<Option<i32>, Box<dyn std::error::Error>> {
        let result = self
            .probe_dtw_with_position(position)
            .expect("Could not probe the position");

        if let Some(val) = result {
            return Ok(Some(val + ply));
        }

        let mut liste = MoveList::new();
        liste.get_moves(position);

        let mut best_value: i32 = 1000;

        for m in liste.iter() {
            let mut copy = position;
            copy.make_move(m);
            let res = self.probe_dtw_with_position(copy).expect("Could not probe");
            let mut val;
            if let Some(v) = res {
                val = v;
            } else {
                val = self
                    .probe_dtw_recursive(copy, orig_mover, ply + 1)
                    .expect("Could not probe")
                    .unwrap_or(1000);
            }
            println!("Value {val}");
            //if orig_mover == pos.color, we are trying to minimize the dtw-value
            //else we are maximizing because we want to prolong the inevitable loss
            val = -val;
            if val < best_value {
                best_value = val;
            }
        }
        Ok(Some(best_value))
    }

    pub fn print_fen(&self, fen_string: &str) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(*const libc::c_char)> =
                self.library.get(b"print_fen")?;
            let c_to_print = CString::new(fen_string).expect("CString failed");
            Ok(func(c_to_print.as_ptr()))
        }
    }

    pub fn get_move_encoding(
        &self,
        previous: &str,
        next: &str,
    ) -> Result<i32, Box<dyn std::error::Error>> {
        unsafe {
            let func: libloading::Symbol<
                unsafe extern "C" fn(*const libc::c_char, *const libc::c_char) -> i32,
            > = self.library.get(b"move_played")?;
            let c_to_previous = CString::new(previous).expect("CString failed");
            let c_to_next = CString::new(next).expect("CString failed");
            Ok(func(c_to_previous.as_ptr(), c_to_next.as_ptr()))
        }
    }
}
