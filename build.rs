extern crate libbindgen;

use std::io::Write;
use std::fs::File;
use std::env;
use std::path::{Path, PathBuf};

fn driver(out_dir: &Path) {
    let bindings = libbindgen::builder()
        .header("/opt/cuda/include/cuda.h")
        .generate()
        .unwrap()
        .to_string();

    File::create(out_dir.join("driver.rs"))
        .unwrap()
        .write(bindings.as_bytes())
        .unwrap();

    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn main() {
    let out_dir = &PathBuf::from(env::var_os("OUT_DIR").unwrap());

    driver(out_dir);
}
