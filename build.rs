extern crate bindgen;

use std::io::Write;
use std::fs::File;
use std::env;
use std::path::{Path, PathBuf};

use bindgen::Builder;

fn driver(out_dir: &Path) {
    let bindings = Builder::new()
        .header("/usr/include/cuda.h")
        .generate()
        .unwrap()
        .to_string();

    File::create(out_dir.join("driver.rs"))
        .unwrap()
        // NOTE(rust-lang/rust#18810) strip `#![allow]`s, these don't work when a source file is
        // `include!`d
        .write(bindings[bindings.find("]").map(|p| p + 1).unwrap_or(0)..].as_bytes())
        .unwrap();

    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn main() {
    let out_dir = &PathBuf::from(env::var_os("OUT_DIR").unwrap());

    driver(out_dir);
}
