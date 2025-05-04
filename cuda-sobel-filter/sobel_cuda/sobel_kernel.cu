// sobel_kernel.cu
#include <torch/extension.h>

__global__ void sobel_kernel(const float* input, float* output, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * W + x;
    if (x >= 1 && x < W-1 && y >= 1 && y < H-1) {
        float gx = -input[(y-1)*W + (x-1)] - 2*input[y*W + (x-1)] - input[(y+1)*W + (x-1)]
                   + input[(y-1)*W + (x+1)] + 2*input[y*W + (x+1)] + input[(y+1)*W + (x+1)];
        float gy = -input[(y-1)*W + (x-1)] - 2*input[(y-1)*W + x] - input[(y-1)*W + (x+1)]
                   + input[(y+1)*W + (x-1)] + 2*input[(y+1)*W + x] + input[(y+1)*W + (x+1)];
        output[idx] = sqrtf(gx * gx + gy * gy);
    }
}

void sobel_cuda(torch::Tensor input, torch::Tensor output) {
    const int H = input.size(0);
    const int W = input.size(1);

    const dim3 threads(16, 16);
    const dim3 blocks((W + 15) / 16, (H + 15) / 16);

    sobel_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), H, W);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sobel_cuda", &sobel_cuda, "Sobel filter (CUDA)");
}
