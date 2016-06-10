#![allow(warnings)]
#![feature(intrinsics)]
#![feature(lang_items)]
#![feature(no_core)]
#![no_core]

use Option::*;
use Ordering::*;

#[no_mangle]
pub fn memcpy_(src: *const f32, dst: *mut f32, n: isize) {
    unsafe {
        let i = overflowing_add(overflowing_mul(block_idx_x(), block_dim_x()), thread_idx_x()) as isize;

        if i < n {
            *(offset(dst, i) as *mut f32) = *offset(src, i)
        }
    }
}

extern "rust-intrinsic" {
    fn block_idx_x() -> i32;
    fn block_dim_x() -> i32;
    fn thread_idx_x() -> i32;

    fn offset<T>(dst: *const T, offset: isize) -> *const T;
    fn overflowing_add<T>(a: T, b: T) -> T;
    fn overflowing_mul<T>(a: T, b: T) -> T;
}

#[lang = "copy"]
trait Copy {}

#[lang = "sized"]
trait Sized {}
// : PartialEq<Rhs>
#[lang = "ord"]
trait PartialOrd<Rhs: ?Sized = Self> {
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;

    #[inline]
    fn lt(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Less) => true,
            _ => false,
        }
    }

    #[inline]
    fn le(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Less) | Some(Equal) => true,
            _ => false,
        }
    }

    #[inline]
    fn gt(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Greater) => true,
            _ => false,
        }
    }

    #[inline]
    fn ge(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Greater) | Some(Equal) => true,
            _ => false,
        }
    }
}

impl PartialOrd for isize {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        loop {}
    }
    fn lt(&self, other: &Self) -> bool { (*self) < (*other) }
    fn le(&self, other: &Self) -> bool { (*self) <= (*other) }
    fn ge(&self, other: &Self) -> bool { (*self) >= (*other) }
    fn gt(&self, other: &Self) -> bool { (*self) > (*other) }
}

enum Option<T> {
    None,
    Some(T),
}

enum Ordering {
    Less = -1,
    Equal = 0,
    Greater = 1,
}
