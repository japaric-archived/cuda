//! CUDA driver
//!
//! Reference: http://docs.nvidia.com/cuda/cuda-driver-api/

use std::ffi::CStr;
use std::marker::PhantomData;
use std::{mem, ptr, result};

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
mod ll;

/// A CUDA "block"
pub struct Block {
    x: u32,
    y: u32,
    z: u32,
}

impl Block {
    /// One dimensional block
    pub fn x(x: u32) -> Self {
        Block { x: x, y: 1, z: 1 }
    }

    /// Two dimensional block
    pub fn xy(x: u32, y: u32) -> Self {
        Block { x: x, y: y, z: 1 }
    }

    /// Three dimensional block
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Block { x: x, y: y, z: z }
    }
}

/// A CUDA "context"
#[derive(Debug)]
pub struct Context {
    defused: bool,
    handle: ll::CUcontext,
}

impl Context {
    // TODO is this actually useful? Note that we are using "RAII" (cf. `drop`)
    // and ownership to manage `Context`es
    #[allow(dead_code)]
    fn current() -> Result<Option<Self>> {
        let mut handle = ptr::null_mut();

        unsafe { lift(ll::cuCtxGetCurrent(&mut handle))? }

        if handle.is_null() {
            Ok(None)
        } else {
            Ok(Some(Context {
                defused: true,
                handle: handle,
            }))
        }
    }

    /// Binds context to the calling thread
    pub fn set_current(&self) -> Result<()> {
        unsafe {
            lift(ll::cuCtxSetCurrent(self.handle))
        }
    }

    /// Loads a PTX module
    pub fn load_module<'ctx>(&'ctx self, image: &CStr) -> Result<Module<'ctx>> {
        let mut handle = ptr::null_mut();

        unsafe {
            lift(ll::cuModuleLoadData(&mut handle, image.as_ptr() as *const _))?
        }

        Ok(Module {
            handle: handle,
            _context: PhantomData,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.defused {
            unsafe { lift(ll::cuCtxDestroy_v2(self.handle)).unwrap() }
        }
    }
}

/// A CUDA device (a GPU)
#[derive(Debug)]
pub struct Device {
    handle: ll::CUdevice,
}

/// Binds to the `nth` device
#[allow(non_snake_case)]
pub fn Device(nth: u16) -> Result<Device> {
    let mut handle = 0;

    unsafe { lift(ll::cuDeviceGet(&mut handle, i32::from(nth)))? }

    Ok(Device { handle: handle })
}

impl Device {
    /// Returns the number of available devices
    pub fn count() -> Result<u32> {
        let mut count: i32 = 0;

        unsafe { lift(ll::cuDeviceGetCount(&mut count))? }

        Ok(count as u32)
    }

    /// Creates a CUDA context for this device
    pub fn create_context(&self) -> Result<Context> {
        let mut handle = ptr::null_mut();
        // TODO expose
        let flags = 0;

        unsafe { lift(ll::cuCtxCreate_v2(&mut handle, flags, self.handle))? }

        Ok(Context {
            defused: false,
            handle: handle,
        })
    }

    /// Returns the maximum number of threads a block can have
    pub fn max_threads_per_block(&self) -> Result<i32> {
        use self::ll::CUdevice_attribute_enum::*;

        self.get(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    /// Returns the total amount of (non necessarily free) memory, in bytes,
    /// that the device has
    pub fn total_memory(&self) -> Result<usize> {
        let mut bytes = 0;

        unsafe { lift(ll::cuDeviceTotalMem_v2(&mut bytes, self.handle))? };

        Ok(bytes)
    }

    fn get(&self, attr: ll::CUdevice_attribute) -> Result<i32> {
        let mut value = 0;

        unsafe {
            lift(ll::cuDeviceGetAttribute(&mut value, attr, self.handle))?
        }

        Ok(value)
    }
}

/// A function that the CUDA device can execute. AKA a "kernel"
pub struct Function<'ctx: 'm, 'm> {
    handle: ll::CUfunction,
    _module: PhantomData<&'m Module<'ctx>>,
}

impl<'ctx, 'm> Function<'ctx, 'm> {
    /// Execute a function on the GPU
    ///
    /// NOTE This function blocks until the GPU has finished executing the
    /// kernel
    pub fn launch(&self,
                  args: &[&Any],
                  grid: Grid,
                  block: Block)
                  -> Result<()> {
        let stream = Stream::new()?;
        // TODO expose
        let shared_mem_bytes = 0;
        // TODO expose
        let extra = ptr::null_mut();

        unsafe {
            lift(ll::cuLaunchKernel(self.handle,
                                    grid.x,
                                    grid.y,
                                    grid.z,
                                    block.x,
                                    block.y,
                                    block.z,
                                    shared_mem_bytes,
                                    stream.handle,
                                    args.as_ptr() as *mut _,
                                    extra))?
        }

        stream.sync()?;
        stream.destroy()
    }
}

/// A CUDA "grid"
pub struct Grid {
    x: u32,
    y: u32,
    z: u32,
}

impl Grid {
    /// One dimensional grid
    pub fn x(x: u32) -> Self {
        Grid { x: x, y: 1, z: 1 }
    }

    /// Two dimensional grid
    pub fn xy(x: u32, y: u32) -> Self {
        Grid { x: x, y: y, z: 1 }
    }

    /// Three dimensional grid
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Grid { x: x, y: y, z: z }
    }
}

/// A PTX module
pub struct Module<'ctx> {
    handle: ll::CUmodule,
    _context: PhantomData<&'ctx Context>,
}

impl<'ctx> Module<'ctx> {
    /// Retrieves a function from the PTX module
    pub fn function<'m>(&'m self, name: &CStr) -> Result<Function<'ctx, 'm>> {
        let mut handle = ptr::null_mut();

        unsafe {
            lift(ll::cuModuleGetFunction(&mut handle,
                                         self.handle,
                                         name.as_ptr()))?
        }

        Ok(Function {
            handle: handle,
            _module: PhantomData,
        })
    }
}

impl<'ctx> Drop for Module<'ctx> {
    fn drop(&mut self) {
        unsafe { lift(ll::cuModuleUnload(self.handle)).unwrap() }
    }
}

// TODO expose
struct Stream {
    handle: ll::CUstream,
}

impl Stream {
    fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // TODO expose
        let flags = 0;

        unsafe { lift(ll::cuStreamCreate(&mut handle, flags))? }

        Ok(Stream { handle: handle })
    }

    fn destroy(self) -> Result<()> {
        unsafe { lift(ll::cuStreamDestroy_v2(self.handle)) }
    }

    fn sync(&self) -> Result<()> {
        unsafe { lift(ll::cuStreamSynchronize(self.handle)) }
    }
}

/// Value who's type has been erased
pub enum Any {}

/// Erase the type of a value
#[allow(non_snake_case)]
pub fn Any<T>(ref_: &T) -> &Any {
    unsafe { &*(ref_ as *const T as *const Any) }
}

/// `memcpy` direction
pub enum Direction {
    /// `src` points to device memory. `dst` points to host memory
    DeviceToHost,
    /// `src` points to host memory. `dst` points to device memory
    HostToDevice,
}

#[allow(missing_docs)]
#[derive(Debug)]
pub enum Error {
    AlreadyAcquired,
    AlreadyMapped,
    ArrayIsMapped,
    Assert,
    ContextAlreadyCurrent,
    ContextAlreadyInUse,
    ContextIsDestroyed,
    Deinitialized,
    EccUncorrectable,
    FileNotFound,
    HardwareStackError,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    IllegalAddress,
    IllegalInstruction,
    InvalidAddressSpace,
    InvalidContext,
    InvalidDevice,
    InvalidGraphicsContext,
    InvalidHandle,
    InvalidImage,
    InvalidPc,
    InvalidPtx,
    InvalidSource,
    InvalidValue,
    LaunchFailed,
    LaunchIncompatibleTexturing,
    LaunchOutOfResources,
    LaunchTimeout,
    MapFailed,
    MisalignedAddress,
    NoBinaryForGpu,
    NoDevice,
    NotFound,
    NotInitialized,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    NotPermitted,
    NotReady,
    NotSupported,
    NvlinkUncorrectable,
    OperatingSystem,
    OutOfMemory,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PeerAccessUnsupported,
    PrimaryContextActive,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    ProfilerDisabled,
    ProfilerNotInitialized,
    SharedObjectInitFailed,
    SharedObjectSymbolNotFound,
    TooManyPeers,
    Unknown,
    UnmapFailed,
    UnsupportedLimit,
}

// TODO should this be a method of `Context`?
/// Allocate `n` bytes of memory on the device
pub unsafe fn allocate(n: usize) -> Result<*mut u8> {
    let mut d_ptr = 0;

    lift(ll::cuMemAlloc_v2(&mut d_ptr, n))?;

    Ok(d_ptr as *mut u8)
}

/// Copy `n` bytes of memory from `src` to `dst`
///
/// `direction` indicates where `src` and `dst` are located (device or host)
pub unsafe fn copy<T>(src: *const T,
                      dst: *mut T,
                      count: usize,
                      direction: Direction)
                      -> Result<()> {
    use self::Direction::*;

    let bytes = count * mem::size_of::<T>();

    lift(match direction {
        DeviceToHost => ll::cuMemcpyDtoH_v2(dst as *mut _, src as u64, bytes),
        HostToDevice => ll::cuMemcpyHtoD_v2(dst as u64, src as *const _, bytes),
    })?;

    Ok(())
}

// TODO same question as `allocate`
/// Free the memory pointed to by `ptr`
pub unsafe fn deallocate(ptr: *mut u8) -> Result<()> {
    lift(ll::cuMemFree_v2(ptr as u64))
}

/// Initialize the CUDA runtime
pub fn initialize() -> Result<()> {
    // TODO expose
    let flags = 0;

    unsafe { lift(ll::cuInit(flags)) }
}

/// Returns the version of the CUDA runtime
pub fn version() -> Result<i32> {
    let mut version = 0;

    unsafe { lift(ll::cuDriverGetVersion(&mut version))? }

    Ok(version)
}

#[allow(missing_docs)]
pub type Result<T> = result::Result<T, Error>;

fn lift(e: ll::CUresult) -> Result<()> {
    use self::Error::*;
    use self::ll::cudaError_enum::*;

    Err(match e {
        CUDA_SUCCESS => return Ok(()),
        CUDA_ERROR_ALREADY_ACQUIRED => AlreadyAcquired,
        CUDA_ERROR_ALREADY_MAPPED => AlreadyMapped,
        CUDA_ERROR_ARRAY_IS_MAPPED => ArrayIsMapped,
        CUDA_ERROR_ASSERT => Assert,
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT => ContextAlreadyCurrent,
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE => ContextAlreadyInUse,
        CUDA_ERROR_CONTEXT_IS_DESTROYED => ContextIsDestroyed,
        CUDA_ERROR_DEINITIALIZED => Deinitialized,
        CUDA_ERROR_ECC_UNCORRECTABLE => EccUncorrectable,
        CUDA_ERROR_FILE_NOT_FOUND => FileNotFound,
        CUDA_ERROR_HARDWARE_STACK_ERROR => HardwareStackError,
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
            HostMemoryAlreadyRegistered
        }
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => HostMemoryNotRegistered,
        CUDA_ERROR_ILLEGAL_ADDRESS => IllegalAddress,
        CUDA_ERROR_ILLEGAL_INSTRUCTION => IllegalInstruction,
        CUDA_ERROR_INVALID_ADDRESS_SPACE => InvalidAddressSpace,
        CUDA_ERROR_INVALID_CONTEXT => InvalidContext,
        CUDA_ERROR_INVALID_DEVICE => InvalidDevice,
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => InvalidGraphicsContext,
        CUDA_ERROR_INVALID_HANDLE => InvalidHandle,
        CUDA_ERROR_INVALID_IMAGE => InvalidImage,
        CUDA_ERROR_INVALID_PC => InvalidPc,
        CUDA_ERROR_INVALID_PTX => InvalidPtx,
        CUDA_ERROR_INVALID_SOURCE => InvalidSource,
        CUDA_ERROR_INVALID_VALUE => InvalidValue,
        CUDA_ERROR_LAUNCH_FAILED => LaunchFailed,
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => LaunchIncompatibleTexturing,
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => LaunchOutOfResources,
        CUDA_ERROR_LAUNCH_TIMEOUT => LaunchTimeout,
        CUDA_ERROR_MAP_FAILED => MapFailed,
        CUDA_ERROR_MISALIGNED_ADDRESS => MisalignedAddress,
        CUDA_ERROR_NOT_FOUND => NotFound,
        CUDA_ERROR_NOT_INITIALIZED => NotInitialized,
        CUDA_ERROR_NOT_MAPPED => NotMapped,
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY => NotMappedAsArray,
        CUDA_ERROR_NOT_MAPPED_AS_POINTER => NotMappedAsPointer,
        CUDA_ERROR_NOT_PERMITTED => NotPermitted,
        CUDA_ERROR_NOT_READY => NotReady,
        CUDA_ERROR_NOT_SUPPORTED => NotSupported,
        CUDA_ERROR_NO_BINARY_FOR_GPU => NoBinaryForGpu,
        CUDA_ERROR_NO_DEVICE => NoDevice,
        CUDA_ERROR_OPERATING_SYSTEM => OperatingSystem,
        CUDA_ERROR_OUT_OF_MEMORY => OutOfMemory,
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => PeerAccessAlreadyEnabled,
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => PeerAccessNotEnabled,
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => PeerAccessUnsupported,
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => PrimaryContextActive,
        CUDA_ERROR_PROFILER_ALREADY_STARTED => ProfilerAlreadyStarted,
        CUDA_ERROR_PROFILER_ALREADY_STOPPED => ProfilerAlreadyStopped,
        CUDA_ERROR_PROFILER_DISABLED => ProfilerDisabled,
        CUDA_ERROR_PROFILER_NOT_INITIALIZED => ProfilerNotInitialized,
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => SharedObjectInitFailed,
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => SharedObjectSymbolNotFound,
        CUDA_ERROR_TOO_MANY_PEERS => TooManyPeers,
        CUDA_ERROR_UNKNOWN => Unknown,
        CUDA_ERROR_UNMAP_FAILED => UnmapFailed,
        CUDA_ERROR_UNSUPPORTED_LIMIT => UnsupportedLimit,
        CUDA_ERROR_NVLINK_UNCORRECTABLE => NvlinkUncorrectable,
    })
}
