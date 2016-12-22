#!/bin/sh

set -ex

bindgen /opt/cuda/include/cuda.h > src/driver/ll.rs
rustfmt src/driver/ll.rs
