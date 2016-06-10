#include <stdint.h>

typedef float f32;
typedef int32_t i32;

extern "C" {
  __global__ void memcpy_(const f32 *src, f32 *dst, i32 n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
      dst[i] = src[i];
    }
  }
}