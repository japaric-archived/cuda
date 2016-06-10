#![cfg(not(debug_assertions))]

extern crate uxx;

macro_rules! test {
    ($($uxx:ident),+) => {
        $(
            mod $uxx {
                use uxx::$uxx;

                #[test]
                fn add() {
                    assert_eq!($uxx(1) + $uxx::max_value(), $uxx(0));
                }

                #[test]
                fn mul() {
                    assert_eq!($uxx::max_value() * $uxx(2), $uxx::max_value() - $uxx(1));
                }

                #[test]
                fn sub() {
                    assert_eq!($uxx(0) - $uxx(1), $uxx::max_value());
                }
            }
        )+
    }
}

test!(u7, u15, u31, u63);
