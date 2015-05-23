# `cuda.rs`

Experiments with CUDA and Rust

## Examples

- The classic [vector addition](/examples/add.rs).
- A RGBA to [grayscale](/examples/gray.rs) image transform.

## Areas to explore

- Generating PTX from Rust code at compile time. ([prior art]).
- Type safety for launching kernels. Arity and argument types should be validated at compile time.
- Linear algebra library with transparent CUDA acceleration. A matrix type that stores its data
  in the GPU, with operator sugar that maps to CuBLAS/custom kernels.
- Kernel creation (at runtime/compile time) from expression templates. Given a expression
  `Z = a * A - b * B + c * C` (lowercase are scalars, uppercase are matrices/vectors), can we
  generate a kernel (a .cu file/string) that performs the operation in one (or minimal amount of)
  memory pass(es). And then evaluate the expression as a single (or a few) kernel call(s).

[linalg]: https://github.com/japaric/linalg.rs
[prior art]: http://blog.theincredibleholk.org/blog/2012/12/05/compiling-rust-for-gpus/

## License

cuda.rs is dual licensed under the Apache 2.0 license and the MIT license.

See LICENSE-APACHE and LICENSE-MIT for more details.
