//! Interface to `nvcc`

use std::ffi::{CString, NulError};
use std::fs::File;
use std::io::{Read, Write, self};
use std::process::Command;

use tempdir::TempDir;

/// `nvcc` builder
#[derive(Default)]
pub struct Nvcc {
    arch: Option<Arch>,
}

impl Nvcc {
    /// Set GPU architecture
    pub fn arch(&mut self, arch: Arch) -> &mut Self {
        self.arch = Some(arch);
        self
    }

    /// Compile `source` code into PTX
    pub fn compile(&self, source: &str) -> Result<CString> {
        let temp_dir = try!(TempDir::new("nvcc"));
        let mut path = temp_dir.path().join("kernel.cu");
        let mut f = try!(File::create(&path));
        try!(f.write_all(source.as_bytes()));
        let mut command = Command::new("nvcc");
        if let Some(ref arch) = self.arch {
            command.arg("-arch").arg(arch.str());
        }
        let output = try!(command.arg("-ptx").arg(&path).current_dir(temp_dir.path()).output());

        if output.status.success() {
            path.set_extension("ptx");
            let mut buf = vec![];
            try!(try!(File::open(&path)).read_to_end(&mut buf));
            Ok(try!(CString::new(buf)))
        } else {
            Err(Error::Stderr(String::from_utf8_lossy(&output.stderr).into_owned()))
        }
    }
}

/// Errors that may raise when calling `nvcc`
#[derive(Debug)]
pub enum Error {
    /// I/O error
    Io(io::Error),
    /// Generated PTX contained a null byte
    Nul(NulError),
    /// `nvcc` call failed
    Stderr(String),
}

impl From<NulError> for Error {
    fn from(e: NulError) -> Error {
        Error::Nul(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::Io(e)
    }
}

/// GPU architecture
#[allow(missing_docs)]
pub enum Arch {
    Compute20,
    Compute30,
    Compute32,
    Compute35,
    Compute37,
    Compute50,
    Compute52,
}

impl Arch {
    fn str(&self) -> &'static str {
        match *self {
            Arch::Compute20 => "compute_20",
            Arch::Compute30 => "compute_30",
            Arch::Compute32 => "compute_32",
            Arch::Compute35 => "compute_35",
            Arch::Compute37 => "compute_37",
            Arch::Compute50 => "compute_50",
            Arch::Compute52 => "compute_52",
        }
    }
}

#[allow(missing_docs)]
pub type Result<T> = ::std::result::Result<T, Error>;
