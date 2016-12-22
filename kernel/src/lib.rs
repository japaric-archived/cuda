#![feature(abi_ptx)]
#![no_std]

extern crate nvptx_builtins as intrinsics;

/// Add two "vectors" of length `n`. `c <- a + b`
#[no_mangle]
pub unsafe extern "ptx-kernel" fn add(a: *const f32,
                                      b: *const f32,
                                      c: *mut f32,
                                      n: usize) {
    let i = (intrinsics::block_dim_x() as isize)
        .wrapping_mul(intrinsics::block_idx_x() as isize)
        .wrapping_add(intrinsics::thread_idx_x() as isize);

    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

/// Copies an array of `n` floating point numbers from `src` to `dst`
#[no_mangle]
pub unsafe extern "ptx-kernel" fn memcpy(dst: *mut f32,
                                         src: *const f32,
                                         n: usize) {
    let i = (intrinsics::block_dim_x() as isize)
        .wrapping_mul(intrinsics::block_idx_x() as isize)
        .wrapping_add(intrinsics::thread_idx_x() as isize);

    if (i as usize) < n {
        *dst.offset(i) = *src.offset(i);
    }
}
