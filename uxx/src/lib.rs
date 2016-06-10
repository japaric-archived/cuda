//! Non-negative signed integers

#![no_std]

#![deny(missing_docs)]
#![deny(warnings)]

extern crate cast;

use core::ops::{Add, Div, Mul, Sub};
use core::fmt;

// TODO add checked casts
// TODO add bunch of other inherent methods: checked ops, etc

macro_rules! uxx {
    ($($uxx:ident: $inner:ident),+) => {
        $(
            /// Non-negative signed integer
            #[allow(non_camel_case_types)]
            #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
            pub struct $uxx {
                inner: $inner,
            }

            /// Constructor
            ///
            /// # Panics
            /// If `x < 0`
            pub fn $uxx(x: $inner) -> $uxx {
                assert!(x >= 0);

                $uxx { inner: x }
            }

            impl $uxx {
                /// Unchecked constructor
                ///
                /// # Panics
                /// If `x < 0` and debug assertions are enabled
                pub unsafe fn unchecked(x: $inner) -> Self {
                    debug_assert!(x >= 0);

                    $uxx { inner: x }
                }

                /// Returns the largest value that can be represented by this integer type
                pub fn max_value() -> Self {
                    $uxx { inner: ::core::$inner::MAX }
                }

                /// Returns the smallest value that can be represented by this integer type
                pub fn min_value() -> Self {
                    $uxx { inner: 0 }
                }
            }

            impl Add for $uxx {
                type Output = Self;

                fn add(self, rhs: Self) -> Self {
                    // NOTE panics on overflow when debug assertions are enabled
                    let result = self.inner + rhs.inner;

                    if result < 0 {
                        $uxx { inner: result & ::core::$inner::MAX }
                    } else {
                        $uxx { inner: result }
                    }
                }
            }

            impl fmt::Debug for $uxx {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    fmt::Debug::fmt(&self.inner, f)
                }
            }

            impl fmt::Display for $uxx {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    fmt::Display::fmt(&self.inner, f)
                }
            }

            impl Div for $uxx {
                type Output = Self;

                fn div(self, rhs: Self) -> Self {
                    // NOTE panics on overflow when debug assertions are enabled
                    let result = self.inner / rhs.inner;

                    $uxx { inner: result }
                }
            }

            impl Mul for $uxx {
                type Output = Self;

                fn mul(self, rhs: Self) -> Self {
                    // NOTE panics on overflow when debug assertions are enabled
                    let result = self.inner * rhs.inner;

                    if result < 0 {
                        $uxx { inner: result & ::core::$inner::MAX }
                    } else {
                        $uxx { inner: result }
                    }
                }
            }

            impl Sub for $uxx {
                type Output = Self;

                fn sub(self, rhs: Self) -> Self {
                    let result = self.inner - rhs.inner;

                    // NOTE panics on overflow
                    debug_assert!(result >= 0, "arithmetic operation overflowed");
                    if result < 0 {
                        $uxx { inner: result & ::core::$inner::MAX }
                    } else {
                        $uxx { inner: result }
                    }
                }
            }
        )+
    }
}

uxx!(u7: i8, u15: i16, u31: i32, u63: i64);

impl cast::From<u31> for i32 {
    type Output = i32;

    fn cast(x: u31) -> i32 {
        x.inner
    }
}
