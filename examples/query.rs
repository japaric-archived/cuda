extern crate cuda;

use cuda::driver::{self, Device};

fn main() {
    run().unwrap();
}

fn run() -> driver::Result<()> {
    try!(driver::initialize());

    println!("Total devices: {}", try!(Device::count()));

    Ok(())
}
