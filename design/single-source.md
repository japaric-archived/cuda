# Single source GPU programs

## What

Extend the `cuda` crate to make it possible to write single source GPU programs.

## Why

The current workflow has several limitations:

- Kernel functions must be written in CUDA and as external `.cu` files. 
- Need to compile kernel functions to PTX at *run-time*.
- Retrieving a kernel function is error prone as it's done in a dictionary lookup fashion using the
  function name (which may be mangled!) as the key.
- Kernel functions don't preserve the type information of their arguments.

Instead:

- We'd like to write the device code in Rust.
- We'd like to compile device code at compile time.
- Retrieving a function from its PTX module should be infallible.
- Kernel functions should preserve their type information.

## What will it look like?

Imagine a `blas` crate that can be compiled to PTX:

``` rust
#![no_std]

fn axpy<T>(alpha: T, x: &[T], y: &mut [T]) {
    // ..
}

// This gets exposed as a kernel function only when this crate is compiled for `nvptx` targets
#[cfg(arch = "nvptx")]
pub extern "cuda-kernel" fn saxpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    axpy(alpha, x, y)
}
```

NOTE: The `blas` crate is not limited to be device code; it can also be compiled for the host
machine.

This `blas` crate can be treated a device code and included in a Rust CUDA program using the
`#[ptx]`/`#[ptx64]` plugin:

``` rust
#[plugin(cuda_plugins)]

#[ptx64]
extern crate blas;

extern crate cuda;

use cuda::{Result, ..};

fn main() {
    run().unwrap_or_else(/* error handling */)
}

fn run() -> Result<()> {
    // ..

    // NOTE: this doesn't involve run-time compile of the device code. The plugin already compiled
    // the device code at compile-time.
    let module = ctx.load_module::<blas::__MODULE__>()?;

    // NOTE: Retrieval doesn't involve (error-prone) strings!
    let kernel = module.function::<blas::saxpy>()?;
    
    // Kernel arguments are type checked!
    kernel.launch((alpha, x, y), grid, block)?;
    
    // This would be a type error -- `y` and `x` has been swapped
    //kernel.launch((alpha, y, x), grid, block);
    //                         ^~ error: expected `&mut [f32]`, got `&[f32]`
    
    // ..
}
```

## How do we do this?

Three steps:

### Convert the `rustc` compiler into a Rust -> PTX compiler

This is WIP. See [1] and [2].

[1]: https://github.com/rust-lang/rfcs/pull/1641
[2]: https://github.com/rust-lang/rust/pull/34195

### Tweak the `cuda` crate

Two traits will be added: `Module` and `Kernel`. These traits encode the idea of a PTX module and a
kernel function respectively. Additionally, the `Kernel` implementer points back to the PTX module
where its contained via an associated type:

``` rust
pub mod traits {
    /// A PTX module
    pub trait Module {
        const PTX: &'static CStr;
    }
    
    /// A kernel function
    pub trait Kernel {
        /// The PTX module where this kernel function resides
        type Module: Module;
        /// The function signature of this kernel
        type Args;
        /// The (maybe mangled) name of the function
        const NAME: &'static CStr;
    }
}
```

The `cuda::driver` module will be updated to use these traits and to provide better type-safety:

``` rust
pub mod driver {
    // Low level bindings to `libcuda.so`
    mod ll {
        // ..
    }

    // NOTE: omitting some ownership semantics (e.g. loaded module can't outlive context, etc) for
    // simplicity
    
    pub struct Context {
        handle: ll::CUcontext,
    }
    
    pub struct Module<M> {
        handle: ll::CUmodule,
        _module: PhantomData<M>,
    }
    
    pub struct Function<K> {
        handle: ll::CUfunction,
        _function: PhantomData<K>,
    }

    impl Context {
        fn load_module<M>(&self) -> Result<Module>
            where M: ::traits::Module,
        {
            let mut handle = ptr::null_mut();
            ll::cuModuleLoadData(&mut handle, M::PTX.as_ptr())?;
            Ok(Module { handle: handle, _module: PhantomData })
        }
    }
    
    impl<M> Module<M> {
        fn function<K>(&self) -> Result<Function<K>>
            // Note the constrain: the kernel `K` must reside in the module `M`
            where K: ::traits::Kernel<Module=M>,
        {
            let mut handle = ptr::null_mut();
            ll::cuModuleGetFunction(&mut handle, self.handle, K::NAME.as_ptr())?;
            Ok(Function { handle: handle, _function: PhantomData })
        }
    }
    
    impl<K> Kernel<K> {
        // Note: passed `args` obey the kernel `K` signature: `K::Args`
        pub fn launch(&self, args: K::Args, block: Block, grid: Grid) -> Result<()> {
            // ..
        }
    }
}
```

### The `ptx`/`ptx64` plugin 

The final piece, the `ptx`/`ptx64` plugin, will take care of expanding a `#[ptx] extern crate foo`
item to a module hierarchy that provides the module and its kernels as types.

Following the previous example, the `#[ptx64] extern crate blas` would expand to:

``` rust
mod blas {
    pub enum __MODULE__ {}
    
    impl ::cuda::traits::Module for __MODULE__ {
        // PTX generated from compiling the `blas` crate for the `nvptx64` target
        const PTX: &'static CStr = /* .. */;
    }
    
    pub enum saxpy {}
    
    impl ::cuda::traits::Kernel for saxpy {
        type Module = __MODULE___;
        // Note: doesn't quite work today...
        type Args = (f32, &[f32], &mut [f32]);
        // Mangled function name
        const NAME: &'static CStr = "_ZN5blas5saxpy17hd3cbcaef4c12c8f8E\0";
    }
}
```

## Unresolved questions

- This design should be compatible with other CUDA features like `__shared__` and `__constant__`
  memory but I haven't thought about it yet.
