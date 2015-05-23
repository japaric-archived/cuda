//! Convert RGB image to grayscale

#![allow(warnings)]

extern crate cast;
extern crate cuda;
extern crate image;
extern crate libc;
extern crate rand;
extern crate time;

use std::{iter, process};

use cast::From;
use cuda::nvcc::{Arch, Nvcc};
use cuda::{Any, Block, Context, Device, Error, Grid, Slice};
use image::{ImageBuffer, Luma};
use rand::{Rng, XorShiftRng};

const RGB_PATH: &'static str = "rgb.jpg";
const GRAY_CPU_PATH: &'static str = "gray_cpu.jpg";
const GRAY_GPU_PATH: &'static str = "gray_gpu.jpg";
const BLOCK_SIZE: u32 = 16;

const KERNEL: &'static str = r#"
#include <stdint.h>

typedef float f32;
typedef uint32_t u32;
typedef uint8_t u8;

struct u8x4 {
    u8 x;
    u8 y;
    u8 z;
    u8 w;
};

extern "C"
__global__
void rgba2gray(const u8x4* rgba, u8* gray, const u32 width, const u32 height) {
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const u32 i = y * width + x;
    const u8 r = rgba[i].x;
    const u8 g = rgba[i].y;
    const u8 b = rgba[i].z;

    gray[i] = u8(.2126 * float(r) + .7152 * float(g) + .0722 * float(b));
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
    let ref kernel = try!(module.get_function("rgba2gray"));
    let elapsed = time::precise_time_ns() - current;
    println!("loading kernel took {} ns", elapsed);

    // open image
    let img = image::open(RGB_PATH).unwrap().to_rgba();
    let (width, height) = img.dimensions();
    let size = usize::from(width * height);
    let rgba_h = &*img;

    // allocate host memory
    let mut gray_h: Vec<u8> = iter::repeat(0).take(size).collect();

    //// allocate device memory
    let current = time::precise_time_ns();
    let mut rgba_d = try!(Slice::<u8>::new(4 * size));
    let mut gray_d = try!(Slice::<u8>::new(size));
    let elapsed = time::precise_time_ns() - current;
    println!("device memory allocation took {} ns", elapsed);

    // copy memory from host to device
    let current = time::precise_time_ns();
    try!(cuda::copy(&rgba_h[..], &mut rgba_d));
    let elapsed = time::precise_time_ns() - current;
    println!("copying memory from host to device took {} ns", elapsed);

    // launch kernel
    let current = time::precise_time_ns();
    let (rgba, gray) = (rgba_d.as_ptr(), gray_d.as_ptr());
    let ref params = [
        Any::new(&rgba),
        Any::new(&gray),
        Any::new(&width),
        Any::new(&height),
    ];
    let grid = Grid::xy((width - 1) / BLOCK_SIZE + 1, (height - 1) / BLOCK_SIZE + 1);
    let block = Block::xy(BLOCK_SIZE, BLOCK_SIZE);
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
    try!(cuda::copy(&gray_d, &mut gray_h[..]));
    let elapsed = time::precise_time_ns() - current;
    println!("copying memory from device to host took {} ns", elapsed);

    println!("RGBA: [{:?} .. {:?}]",
             rgba_h.chunks(4).next().unwrap(), rgba_h.chunks(4).last().unwrap());
    println!("Gray: [{} .. {}]", gray_h[0], gray_h.last().unwrap());

    {
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, &*gray_h).unwrap();
        img.save(GRAY_GPU_PATH).unwrap();
    }

    // transform image in the CPU
    let current = time::precise_time_ns();
    for i in 0..size {
        let r = f32::from(rgba_h[4*i]);
        let g = f32::from(rgba_h[4*i + 1]);
        let b = f32::from(rgba_h[4*i + 2]);

        gray_h[i] = u8::from(0.2126 * r + 0.7152 * g + 0.0722 * b).unwrap();
    }
    let elapsed = time::precise_time_ns() - current;
    println!("transforming image in the CPU took {} ns", elapsed);

    {
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, &*gray_h).unwrap();
        img.save(GRAY_CPU_PATH).unwrap();
    }

    Ok(())
}

fn main() {
    run().unwrap_or_else(|e| {
        println!("error: {:?}", e);
        process::exit(1)
    })
}
