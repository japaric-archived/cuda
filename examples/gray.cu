extern "C" {
__global__ void rgba_to_grayscale(const uchar4 *const rgbaImage,
                                  unsigned char *const greyImage, int width,
                                  int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= width || y >= height) {
    return;
  }

  int i = y * width + x;
  uchar4 pixel = rgbaImage[i];
  unsigned char red = pixel.x;
  unsigned char green = pixel.y;
  unsigned char blue = pixel.z;

  greyImage[i] = .299 * red + .587 * green + .114 * blue;
}
}