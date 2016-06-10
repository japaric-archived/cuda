extern crate cast;
extern crate cuda;
extern crate hyper;
extern crate image;
extern crate uxx;

use std::mem;
use std::ffi::CStr;
use std::io::Read;
use std::os::raw::c_int;
use std::path::Path;

use cast::From as _0;
use cast::usize;
use cuda::compile;
use cuda::driver::{self, Any, Block, Device, Direction, Grid};
use hyper::Client;
use hyper::header::Connection;
use image::{ColorType, DynamicImage, ImageFormat, Rgba};
use uxx::u31;

const URL: &'static str = "http://www.rustacean.net/assets/rustacean-orig-noshadow.png";
const KERNEL: &'static str = include_str!("gray.cu");

fn main() {
    run().unwrap();
}

fn run() -> Result<(), driver::Error> {
    try!(driver::initialize());
    let dev = try!(Device(u31(0)));
    let ctx = try!(dev.create_context());
    let mod_ = try!(ctx.load_module(&compile::source(KERNEL).unwrap()));
    let kernel = try!(mod_.function(CStr::from_bytes_with_nul(b"rgba_to_grayscale\0").unwrap()));

    let di = fetch();
    let rgba_img = di.as_rgba8().unwrap();
    let (w, h) = rgba_img.dimensions();
    let npixels = usize(w) * usize(h);
    let rgba_bytes = npixels * mem::size_of::<Rgba<u8>>();
    let h_rgba = rgba_img.as_ptr();

    // Allocate memory on device
    let (d_rgba, d_gray) =
        unsafe { (try!(driver::allocate(rgba_bytes)), try!(driver::allocate(npixels))) };

    // Memcpy Host -> Device
    unsafe {
        try!(driver::copy(h_rgba, d_rgba, rgba_bytes, Direction::HostToDevice));
    }

    // Launch kernel
    try!(kernel.launch(&[Any(&d_rgba),
                         Any(&d_gray),
                         Any(&c_int::cast(w).unwrap()),
                         Any(&c_int::cast(h).unwrap())],
                       Grid::xy(32, 32),
                       Block::xy((w + w % 32) / 32, (h + h % 32) / 32)));

    // Memcpy Device -> Host
    let h_gray = &mut unsafe {
        let mut v = Vec::<u8>::with_capacity(npixels);
        v.set_len(npixels);
        v
    };
    unsafe {
        try!(driver::copy(d_gray,
                          h_gray.as_mut_ptr(),
                          npixels,
                          Direction::DeviceToHost));
    }

    // Free memory on device
    unsafe {
        try!(driver::deallocate(d_rgba as *mut _));
        try!(driver::deallocate(d_gray as *mut _));
    }

    image::save_buffer(Path::new("rgba.png"), rgba_img, w, h, ColorType::RGBA(8)).unwrap();
    image::save_buffer(Path::new("gray.device.png"),
                       h_gray,
                       w,
                       h,
                       ColorType::Gray(8))
        .unwrap();

    Ok(())
}

fn fetch() -> DynamicImage {
    let client = Client::new();

    let res = &mut client.get(URL).header(Connection::close()).send().unwrap();

    assert_eq!(res.status, hyper::Ok);

    let buf = &mut vec![];
    res.read_to_end(buf).unwrap();
    image::load_from_memory_with_format(buf, ImageFormat::PNG).unwrap()
}
