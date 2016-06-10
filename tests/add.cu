#include <stdint.h>

typedef float f32;
typedef uint32_t u32;

extern "C" {
__global__ void add(const f32 *a, const f32 *b, f32 *c, u32 n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
}