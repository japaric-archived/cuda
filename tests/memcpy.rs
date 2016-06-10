extern crate cuda;
extern crate rand;
extern crate uxx;

use std::ffi::CStr;
use std::mem;

use cuda::compile;
use cuda::driver::{self, Any, Block, Device, Direction, Grid, Result};
use rand::{Rng, XorShiftRng};
use uxx::u31;

const KERNEL: &'static str = include_str!("memcpy.cu");

#[test]
fn memcpy() {
    run().unwrap();
}

fn run() -> Result<()> {
    const SIZE: usize = 1024 * 1024;

    // Compile KERNEL
    let ref ptx = compile::source(KERNEL).unwrap();

    // Allocate memory on host
    let ref mut rng: XorShiftRng = rand::thread_rng().gen();

    let ref h_a = (0..SIZE).map(|_| rng.gen()).collect::<Vec<f32>>();
    let ref mut h_b = (0..SIZE).map(|_| 0.).collect::<Vec<f32>>();

    // Initialize driver, and load kernel
    try!(driver::initialize());

    let device = try!(Device(u31(0)));
    let ctx = try!(device.create_context());
    let module = try!(ctx.load_module(ptx));
    let kernel = try!(module.function(&CStr::from_bytes_with_nul(b"memcpy_\0").unwrap()));

    // Allocate memory on device
    let (d_a, d_b) = unsafe {
        let bytes = SIZE * mem::size_of::<f32>();

        (try!(driver::allocate(bytes)) as *mut f32, try!(driver::allocate(bytes)) as *mut f32)
    };

    // Memcpy Host -> Device
    unsafe {
        try!(driver::copy(h_a.as_ptr(), d_a, SIZE, Direction::HostToDevice));
    }

    // Launch kernel
    let n = SIZE as u32;
    let nthreads = try!(device.max_threads_per_block()) as u32;
    let nblocks = n / nthreads;
    try!(kernel.launch(&[Any(&d_a), Any(&d_b), Any(&n)],
                       Grid::x(nblocks),
                       Block::x(nthreads)));

    // Memcpy device -> host
    unsafe {
        try!(driver::copy(d_b, h_b.as_mut_ptr(), SIZE, Direction::DeviceToHost));
    }

    // Free memory on device
    unsafe {
        try!(driver::deallocate(d_a as *mut _));
        try!(driver::deallocate(d_b as *mut _));
    }

    // Check results match
    assert_eq!(h_a, h_b);

    Ok(())
}
