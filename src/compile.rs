use std::ffi::{CString, NulError};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::Command;
use std::result;

use tempdir::TempDir;

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// `nvcc` error, this variant contains `nvcc`'s `stderr`
    Compile(String),
    /// I/O error
    Io(io::Error),
    /// `nvcc` output contained a nul byte (this would be a bug AFAIK)
    Nul(NulError),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::Io(e)
    }
}

impl From<NulError> for Error {
    fn from(e: NulError) -> Error {
        Error::Nul(e)
    }
}

fn common(temp_dir: &Path, source: &Path) -> Result<CString> {
    let mut buffer = String::new();

    let output = try!(Command::new("nvcc")
        .args(&["-ptx", "-o", "output.ptx"])
        .arg(source)
        .current_dir(temp_dir)
        .output());

    if !output.status.success() {
        return Err(Error::Compile(String::from_utf8(output.stderr).unwrap()));
    }

    try!(try!(File::open(temp_dir.join("output.ptx"))).read_to_string(&mut buffer));

    Ok(try!(CString::new(buffer)))
}

pub fn file<P>(path: P) -> Result<CString>
    where P: AsRef<Path>
{
    file_(&try!(path.as_ref().canonicalize()))
}

fn file_(path: &Path) -> Result<CString> {
    let temp_dir = &try!(TempDir::new("nvcc"));

    common(temp_dir.path(), path)
}

pub fn source(source: &str) -> Result<CString> {
    let temp_dir = &try!(TempDir::new("nvcc"));
    let temp_dir = temp_dir.path();

    try!(try!(File::create(temp_dir.join("input.cu"))).write_all(source.as_bytes()));

    common(temp_dir, Path::new("input.cu"))
}
