//! Vector addition

extern crate cast;
extern crate cuda;
extern crate libc;
extern crate rand;
extern crate time;

use std::process;

use cast::From;
use cuda::nvcc::{Arch, Nvcc};
use cuda::{Any, Block, Context, Device, Error, Grid, Slice};
use rand::{Rng, XorShiftRng};

const SIZE: u32 = 1024 * 1024;
const THREADS_PER_BLOCK: u32 = 1024;

/// Kernel for vector addition
const KERNEL: &'static str = r#"
#include <stdint.h>

typedef float f32;
typedef uint32_t u32;

extern "C"
__global__
void add(const f32 *A, const f32 *B, f32 *C, const u32 n) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    C[i] = A[i] + B[i];
}
"#;

fn run() -> Result<(), Error> {
    // initialization
    try!(cuda::init());
    let dev = try!(Device::get(0));
    let ctx = try!(Context::new(&dev));
    let stream = try!(ctx.stream());

    // compile kernel
    let current = time::precise_time_ns();
    let ptx = Nvcc::default().arch(Arch::Compute30).compile(KERNEL).unwrap();
    let elapsed = time::precise_time_ns() - current;
    println!("compiling kernel took {} ns", elapsed);

    // load kernel
    let current = time::precise_time_ns();
    let module = try!(ctx.load(&ptx));
    let ref kernel = try!(module.get_function("add"));
    let elapsed = time::precise_time_ns() - current;
    println!("loading kernel took {} ns", elapsed);

    // initialize vectors
    let size = usize::from(SIZE);
    let ref mut rng: XorShiftRng = rand::thread_rng().gen();
    let a_h: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
    let b_h: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
    let mut c_h: Vec<_> = (0..size).map(|_| 0.).collect();

    // allocate device memory
    let current = time::precise_time_ns();
    let mut a_d = try!(Slice::<f32>::new(size));
    let mut b_d = try!(Slice::<f32>::new(size));
    let c_d = try!(Slice::<f32>::new(size));
    let elapsed = time::precise_time_ns() - current;
    println!("device memory allocation took {} ns", elapsed);

    // copy memory from host to device
    let current = time::precise_time_ns();
    try!(cuda::copy(&a_h[..], &mut a_d));
    try!(cuda::copy(&b_h[..], &mut b_d));
    let elapsed = time::precise_time_ns() - current;
    println!("copying memory from host to device took {} ns", elapsed);

    // launch kernel
    let current = time::precise_time_ns();
    let (a, b, c, n) = (a_d.as_ptr(), b_d.as_ptr(), c_d.as_ptr(), SIZE);
    let ref params = [
        Any::new(&a),
        Any::new(&b),
        Any::new(&c),
        Any::new(&n),
    ];
    let grid = Grid::x(SIZE / THREADS_PER_BLOCK);
    let block = Block::x(THREADS_PER_BLOCK);
    try!(stream.launch(kernel, grid, block, 0, params));
    let elapsed = time::precise_time_ns() - current;
    println!("launching kernel took {} ns", elapsed);

    // wait until kernel finishes
    let current = time::precise_time_ns();
    try!(stream.sync());
    let elapsed = time::precise_time_ns() - current;
    println!("executing kernel took {} ns", elapsed);

    // copy memory from device to host
    let current = time::precise_time_ns();
    try!(cuda::copy(&c_d, &mut c_h[..]));
    let elapsed = time::precise_time_ns() - current;
    println!("copying memory from device to host took {} ns", elapsed);

    println!("A: [{:.4} {:.4} .. {:.4} {:.4}]", a_h[0], a_h[1], a_h[size-2], a_h[size-1]);
    println!("B: [{:.4} {:.4} .. {:.4} {:.4}]", b_h[0], b_h[1], b_h[size-2], b_h[size-1]);
    println!("C: [{:.4} {:.4} .. {:.4} {:.4}]", c_h[0], c_h[1], c_h[size-2], c_h[size-1]);

    let current = time::precise_time_ns();
    for i in 0..size {
        c_h[i] = a_h[i] + b_h[i];
    }
    let elapsed = time::precise_time_ns() - current;
    println!("adding elements sequentially in the CPU took {} ns", elapsed);

    Ok(())
}

fn main() {
    run().unwrap_or_else(|e| {
        println!("error: {:?}", e);
        process::exit(1)
    })
}
