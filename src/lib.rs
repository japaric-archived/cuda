//! Experiments with CUDA and Rust

#![deny(missing_docs)]
#![deny(warnings)]
#![feature(plugin)]
#![plugin(bindgen_plugin)]

extern crate cast;
extern crate libc;
extern crate tempdir;

mod ll;

pub mod nvcc;

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::{mem, ptr};

use cast::From;
use libc::{c_int, size_t};

/// Anything! No type information, not even at runtime
pub enum Any {}

impl Any {
    /// Erases the type `T` and creates a pointer to anything
    pub fn new<'a, T>(thing: &'a T) -> &'a Any where T: Sized {
        unsafe {
            mem::transmute(thing)
        }
    }
}

/// A thread block
pub struct Block {
    x: u32,
    y: u32,
    z: u32
}

impl Block {
    /// One dimensional block with `x` threads
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `x > 1024`
    pub fn x(x: u32) -> Block {
        assert!(x <= 1024);

        Block {
            x: x,
            y: 1,
            z: 1,
        }
    }

    /// Two dimensional block with `x * y` threads
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `x > 1024 ||`
    /// - `y > 1024`
    pub fn xy(x: u32, y: u32) -> Block {
        assert!(x <= 1024);
        assert!(y <= 1024);
        assert!(x * y <= 1024);

        Block {
            x: x,
            y: y,
            z: 1,
        }
    }

    /// Three dimensional block with `x * y * z` threads
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `x > 1024 ||`
    /// - `y > 1024 ||`
    /// - `y > 64`
    pub fn xyz(x: u32, y: u32, z: u32) -> Block {
        assert!(x <= 1024);
        assert!(y <= 1024);
        assert!(z <= 64);
        assert!(x * y * z <= 1024);

        Block {
            x: x,
            y: y,
            z: z,
        }
    }
}

/// A CUDA context
pub struct Context(ll::CUcontext);

impl Context {
    /// Creates a new CUDA context for `device`
    pub fn new(device: &Device) -> Result<Context> {
        unsafe {
            let flags = 0;
            let mut pctx = ptr::null_mut();

            match ll::cuCtxCreate_v2(&mut pctx, flags, device.0) {
                ll::CUDA_SUCCESS => Ok(Context(pctx)),
                code => Err(Error::from(code)),
            }
        }
    }

    /// Loads PTX into a compute module
    pub fn load<'ctx>(&'ctx self, ptx: &CStr) -> Result<Module<'ctx>> {
        unsafe {
            let mut module = ptr::null_mut();

            match ll::cuModuleLoadData(&mut module, ptx.as_ptr() as *const _) {
                ll::CUDA_SUCCESS => Ok(Module(module, PhantomData)),
                code => Err(Error::from(code)),
            }
        }
    }

    /// Creates a new stream
    pub fn stream<'ctx>(&'ctx self) -> Result<Stream<'ctx>> {
        unsafe {
            let mut stream = ptr::null_mut();
            let flags = 0;

            match ll::cuStreamCreate(&mut stream, flags) {
                ll::CUDA_SUCCESS => Ok(Stream(stream, PhantomData)),
                code => Err(Error::from(code)),
            }
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let code = ll::cuCtxDestroy_v2(self.0);

            debug_assert_eq!(code, ll::CUDA_SUCCESS);
        }
    }
}

/// A compute device
pub struct Device(ll::CUdevice);

impl Device {
    /// Returns the number of compute-capable devices
    pub fn count() -> Result<u32> {
        unsafe {
            let mut n = 0;

            match ll::cuDeviceGetCount(&mut n) {
                ll::CUDA_SUCCESS => Ok(u32::from(n).unwrap()),
                code => Err(Error::from(code)),
            }
        }
    }

    /// Returns a handle to the `i`th device
    pub fn get(i: u32) -> Result<Device> {
        unsafe {
            let mut device = 0;

            match ll::cuDeviceGet(&mut device, c_int::from(i).unwrap()) {
                ll::CUDA_SUCCESS => Ok(Device(device)),
                code => Err(Error::from(code)),
            }
        }
    }
}

/// A CUDA function
pub struct Function<'ctx>(ll::CUfunction, PhantomData<&'ctx Context>);

/// A grid of blocks
pub struct Grid {
    x: u32,
    y: u32,
    z: u32,
}

impl Grid {
    /// One dimensional grid with `x` blocks
    pub fn x(x: u32) -> Grid {
        assert!(x < 1 << 31);

        Grid {
            x: x,
            y: 1,
            z: 1,
        }
    }

    /// Two dimensional grid with `x * y` blocks
    pub fn xy(x: u32, y: u32) -> Grid {
        assert!(x < 1 << 31);
        assert!(y < 65536);

        Grid {
            x: x,
            y: y,
            z: 1,
        }
    }

    /// Three dimensional grid with `x * y * z` blocks
    pub fn xyz(x: u32, y: u32, z: u32) -> Grid {
        assert!(x < 1 << 31);
        assert!(y < 65536);
        assert!(z < 65536);

        Grid {
            x: x,
            y: y,
            z: z,
        }
    }
}

/// A compute module
pub struct Module<'ctx>(ll::CUmodule, PhantomData<&'ctx Context>);

impl<'ctx> Module<'ctx> {
    /// Retrieves a function from this module
    pub fn get_function(&self, name: &str) -> Result<Function<'ctx>> {
        unsafe {
            let mut func = ptr::null_mut();
            let cstring = CString::new(name).unwrap();

            match ll::cuModuleGetFunction(&mut func, self.0, cstring.as_ptr() as *const _) {
                ll::CUDA_SUCCESS => Ok(Function(func, PhantomData)),
                code => Err(Error::from(code)),
            }
        }
    }
}

impl<'ctx> Drop for Module<'ctx> {
    fn drop(&mut self) {
        unsafe {
            let code = ll::cuModuleUnload(self.0);

            debug_assert_eq!(code, ll::CUDA_SUCCESS);
        }
    }
}

/// A slice of memory owned by the GPU
pub struct Slice<T> {
    _marker: PhantomData<*mut T>,
    data: ll::CUdeviceptr,
    len: usize,
}

impl<T> Slice<T> {
    /// Allocates a slice of size `len` on the GPU
    pub fn new(len: usize) -> Result<Slice<T>> where T: Copy {
        unsafe {
            let mut data = 0;
            let bytesize = size_t::from(len * mem::size_of::<T>());

            match ll::cuMemAlloc_v2(&mut data, bytesize) {
                ll::CUDA_SUCCESS => Ok(Slice { _marker: PhantomData, data: data, len: len }),
                code => Err(Error::from(code)),
            }
        }
    }

    /// Returns a pointer to the start of the device memory
    pub fn as_ptr(&self) -> ll::CUdeviceptr {
        ll::CUdeviceptr::from(self.data)
    }

    fn bytecount(&self) -> size_t {
        size_t::from(self.len() * mem::size_of::<T>())
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for Slice<T> {
    fn drop(&mut self) {
        unsafe {
            let code = ll::cuMemFree_v2(ll::CUdeviceptr::from(self.data));

            debug_assert_eq!(code, 0);
        }
    }
}

/// A stream of tasks
pub struct Stream<'ctx>(ll::CUstream, PhantomData<&'ctx Context>);

impl<'ctx> Stream<'ctx> {
    /// Enqueues a kernel in this stream
    // XXX(japaric) `params` should remain valid until the kernel finishes. This seems hard to
    // encode in the lifetimes.
    pub fn launch(
        &self,
        kernel: &Function,
        grid: Grid,
        block: Block,
        shared_mem_bytes: u32,
        params: &[&Any],
    ) -> Result<()> {
        unsafe {
            let extra = ptr::null_mut();
            let params = params.as_ptr() as *mut _;
            let stream = self.0;

            match ll::cuLaunchKernel(
                kernel.0, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_mem_bytes,
                stream, params, extra)
            {
                ll::CUDA_SUCCESS => Ok(()),
                code => Err(Error::from(code)),
            }
        }
    }

    /// Blocks until all tasks are completed
    pub fn sync(&self) -> Result<()> {
        unsafe {
            match ll::cuStreamSynchronize(self.0) {
                ll::CUDA_SUCCESS => Ok(()),
                code => Err(Error::from(code)),
            }
        }
    }
}

impl<'ctx> Drop for Stream<'ctx> {
    fn drop(&mut self) {
        unsafe {
            let code = ll::cuStreamDestroy_v2(self.0);

            debug_assert_eq!(code, ll::CUDA_SUCCESS);
        }
    }
}

/// Error
#[allow(missing_docs)]
#[derive(Debug)]
pub enum Error {
    InvalidContext,
    InvalidDevice,
    InvalidValue,
    NoBinaryForGpu,
    NotFound,
    NotInitialized,
}

impl Error {
    fn from(code: ll::CUresult) -> Error {
        match code {
            ll::CUDA_ERROR_INVALID_CONTEXT => Error::InvalidContext,
            ll::CUDA_ERROR_INVALID_DEVICE => Error::InvalidDevice,
            ll::CUDA_ERROR_INVALID_VALUE => Error::InvalidValue,
            ll::CUDA_ERROR_NOT_FOUND => Error::NotFound,
            ll::CUDA_ERROR_NOT_INITIALIZED => Error::NotInitialized,
            ll::CUDA_ERROR_NO_BINARY_FOR_GPU => Error::NoBinaryForGpu,
            code => panic!("CUresult {} not yet implemented", code),
        }
    }
}

/// `Memcpy`able data
pub trait Memcpy<Src: ?Sized> {
    /// Copies data from `src` into `self`
    fn memcpy(&mut self, &Src) -> Result<()>;
}

impl<T> Memcpy<[T]> for Slice<T> {
    fn memcpy(&mut self, src: &[T]) -> Result<()> {
        unsafe {
            assert_eq!(self.len(), src.len());

            let byte_count = self.bytecount();
            let dst = self.as_ptr();
            let src = src.as_ptr() as *const _;

            match ll::cuMemcpyHtoD_v2(dst, src, byte_count) {
                ll::CUDA_SUCCESS => Ok(()),
                code => Err(Error::from(code)),
            }
        }
    }
}

impl<T> Memcpy<Slice<T>> for [T] {
    fn memcpy(&mut self, src: &Slice<T>) -> Result<()> {
        unsafe {
            assert_eq!(self.len(), src.len());

            let byte_count = src.bytecount();
            let dst = self.as_mut_ptr() as *mut _;
            let src = src.as_ptr();

            match ll::cuMemcpyDtoH_v2(dst, src, byte_count) {
                ll::CUDA_SUCCESS => Ok(()),
                code => Err(Error::from(code)),
            }
        }
    }
}

/// Copies data from `src` to `dst`
///
/// # Panics
///
/// if dimensions don't match
pub fn copy<S: ?Sized, D: ?Sized>(src: &S, dst: &mut D) -> Result<()> where D: Memcpy<S> {
    dst.memcpy(src)
}

/// Initialize the CUDA API
pub fn init() -> Result<()> {
    unsafe {
        match ll::cuInit(0) {
            ll::CUDA_SUCCESS => Ok(()),
            code => Err(Error::from(code)),
        }
    }
}

#[allow(missing_docs)]
pub type Result<T> = ::std::result::Result<T, Error>;
