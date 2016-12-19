use std::ffi::CStr;
use std::marker::PhantomData;
use std::{mem, ptr, result};

use cast::i32;
use uxx::u31;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[cfg_attr(clippy, allow(expl_impl_clone_on_copy))]
#[cfg_attr(clippy, allow(used_underscore_binding))]
#[cfg_attr(rustfmt, rustfmt_skip)]
mod ll;

pub struct Block {
    x: u32,
    y: u32,
    z: u32,
}

impl Block {
    pub fn x(x: u32) -> Self {
        Block { x: x, y: 1, z: 1 }
    }

    pub fn xy(x: u32, y: u32) -> Self {
        Block { x: x, y: y, z: 1 }
    }

    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Block { x: x, y: y, z: z }
    }
}

#[derive(Debug)]
pub struct Context {
    defused: bool,
    handle: ll::CUcontext,
}

impl Context {
    pub fn current() -> Result<Option<Self>> {
        let mut handle = ptr::null_mut();

        unsafe { try!(lift(ll::cuCtxGetCurrent(&mut handle))) }

        if handle.is_null() {
            Ok(None)
        } else {
            Ok(Some(Context {
                defused: true,
                handle: handle,
            }))
        }
    }

    pub fn load_module<'ctx>(&'ctx self, image: &CStr) -> Result<Module<'ctx>> {
        let mut handle = ptr::null_mut();

        unsafe { try!(lift(ll::cuModuleLoadData(&mut handle, image.as_ptr() as *const _))) }

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

#[derive(Debug)]
pub struct Device {
    handle: ll::CUdevice,
}

#[allow(non_snake_case)]
pub fn Device(ordinal: u31) -> Result<Device> {
    let mut handle = 0;

    unsafe { try!(lift(ll::cuDeviceGet(&mut handle, i32(ordinal)))) }

    Ok(Device { handle: handle })
}

impl Device {
    pub fn count() -> Result<u31> {
        let mut count: i32 = 0;

        unsafe { try!(lift(ll::cuDeviceGetCount(&mut count))) }

        Ok(unsafe { u31::unchecked(count) })
    }

    pub fn create_context(&self) -> Result<Context> {
        let mut handle = ptr::null_mut();
        // TODO expose
        let flags = 0;

        unsafe { try!(lift(ll::cuCtxCreate_v2(&mut handle, flags, self.handle))) }

        Ok(Context {
            defused: false,
            handle: handle,
        })
    }

    pub fn max_threads_per_block(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut bytes = 0;

        unsafe { try!(lift(ll::cuDeviceTotalMem_v2(&mut bytes, self.handle))) };

        Ok(bytes)
    }

    fn get(&self, attr: ll::CUdevice_attribute) -> Result<i32> {
        let mut value = 0;

        unsafe { try!(lift(ll::cuDeviceGetAttribute(&mut value, attr, self.handle))) }

        Ok(value)
    }
}

pub struct Function<'ctx: 'm, 'm> {
    handle: ll::CUfunction,
    _module: PhantomData<&'m Module<'ctx>>,
}

impl<'ctx, 'm> Function<'ctx, 'm> {
    pub fn launch(&self, args: &[&Any], grid: Grid, block: Block) -> Result<()> {
        let stream = try!(Stream::new());
        // TODO expose
        let shared_mem_bytes = 0;
        // TODO expose
        let extra = ptr::null_mut();

        unsafe {
            try!(lift(ll::cuLaunchKernel(self.handle,
                                         grid.x,
                                         grid.y,
                                         grid.z,
                                         block.x,
                                         block.y,
                                         block.z,
                                         shared_mem_bytes,
                                         stream.handle,
                                         args.as_ptr() as *mut _,
                                         extra)))
        }

        try!(stream.sync());
        stream.destroy()
    }
}

pub struct Grid {
    x: u32,
    y: u32,
    z: u32,
}

impl Grid {
    pub fn x(x: u32) -> Self {
        Grid { x: x, y: 1, z: 1 }
    }

    pub fn xy(x: u32, y: u32) -> Self {
        Grid { x: x, y: y, z: 1 }
    }

    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Grid { x: x, y: y, z: z }
    }
}

pub struct Module<'ctx> {
    handle: ll::CUmodule,
    _context: PhantomData<&'ctx Context>,
}

impl<'ctx> Module<'ctx> {
    pub fn function<'m>(&'m self, name: &CStr) -> Result<Function<'ctx, 'm>> {
        let mut handle = ptr::null_mut();

        unsafe { try!(lift(ll::cuModuleGetFunction(&mut handle, self.handle, name.as_ptr()))) }

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

        unsafe { try!(lift(ll::cuStreamCreate(&mut handle, flags))) }

        Ok(Stream { handle: handle })
    }

    fn destroy(self) -> Result<()> {
        unsafe { lift(ll::cuStreamDestroy_v2(self.handle)) }
    }

    fn sync(&self) -> Result<()> {
        unsafe { lift(ll::cuStreamSynchronize(self.handle)) }
    }
}

pub enum Any {}

#[allow(non_snake_case)]
pub fn Any<T>(ref_: &T) -> &Any {
    unsafe { &*(ref_ as *const T as *const Any) }
}

pub enum Direction {
    DeviceToHost,
    HostToDevice,
}

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

pub unsafe fn allocate(size: usize) -> Result<*mut u8> {
    let mut dptr = 0;

    try!(lift(ll::cuMemAlloc_v2(&mut dptr, size)));

    Ok(dptr as *mut u8)
}

pub unsafe fn copy<T>(src: *const T,
                      dst: *mut T,
                      count: usize,
                      direction: Direction)
                      -> Result<()> {
    use self::Direction::*;

    let bytes = count * mem::size_of::<T>();

    try!(lift(match direction {
        DeviceToHost => ll::cuMemcpyDtoH_v2(dst as *mut _, src as u64, bytes),
        HostToDevice => ll::cuMemcpyHtoD_v2(dst as u64, src as *const _, bytes),
    }));

    Ok(())
}

pub unsafe fn deallocate(ptr: *mut u8) -> Result<()> {
    lift(ll::cuMemFree_v2(ptr as u64))
}

pub fn initialize() -> Result<()> {
    // TODO expose
    let flags = 0;

    unsafe { lift(ll::cuInit(flags)) }
}

pub fn version() -> Result<i32> {
    let mut version = 0;

    unsafe { try!(lift(ll::cuDriverGetVersion(&mut version))) }

    Ok(version)
}

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
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => HostMemoryAlreadyRegistered,
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
