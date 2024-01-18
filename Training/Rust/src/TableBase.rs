use crate::Sample;
use libc;
use libloading;
use std::ffi::CString;
use std::os::raw::c_char;
pub struct Base<'a> {
    path: &'a str,
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

impl<'a> Base<'a> {
    pub fn new(
        path: &str,
        cache_size: i32,
        num_pieces: i32,
    ) -> Result<Base<'_>, Box<dyn std::error::Error>> {
        unsafe {
            let base = Base {
                path,
                cache_size,
                num_pieces,
                library: libloading::Library::new("libRustDll.dll")?,
            };
            let func: libloading::Symbol<
                unsafe extern "C" fn(*const libc::c_char, libc::c_int, libc::c_int),
            > = base.library.get(b"load")?;
            let c_to_print = CString::new(base.path).expect("CString failed");
            func(c_to_print.as_ptr(), base.cache_size, base.num_pieces);
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

    pub fn print_fen(&self, fen_string: &str) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(*const libc::c_char)> =
                self.library.get(b"print_fen")?;
            let c_to_print = CString::new(fen_string).expect("CString failed");
            Ok(func(c_to_print.as_ptr()))
        }
    }
}
