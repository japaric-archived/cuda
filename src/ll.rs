#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[link(name = "cuda")]
extern {}

bindgen!("/usr/include/cuda.h");
