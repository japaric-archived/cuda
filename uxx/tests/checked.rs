extern crate uxx;

macro_rules! test {
    ($($uxx:ident),+) => {
        $(
            mod $uxx {
                use uxx::$uxx;

                #[should_panic]
                #[test]
                fn constructor() {
                    $uxx(-1);
                }
            }
        )+
    }
}

test!(u7, u15, u31, u63);
