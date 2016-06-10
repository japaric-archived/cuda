#![cfg(debug_assertions)]

extern crate uxx;

macro_rules! test {
    ($($uxx:ident),+) => {
        $(
            mod $uxx {
                use uxx::$uxx;

                #[should_panic]
                #[test]
                fn add() {
                    $uxx(1) + $uxx::max_value();
                }

                #[should_panic]
                #[test]
                fn div() {
                    $uxx(1) / $uxx(0);
                }

                #[should_panic]
                #[test]
                fn mul() {
                    $uxx::max_value() * $uxx(2);
                }

                #[should_panic]
                #[test]
                fn sub() {
                    $uxx(0) - $uxx(1);
                }

                #[should_panic]
                #[test]
                fn unchecked() {
                    unsafe {
                        $uxx::unchecked(-1);
                    }
                }
            }
        )+
    }
}

test!(u7, u15, u31, u63);
