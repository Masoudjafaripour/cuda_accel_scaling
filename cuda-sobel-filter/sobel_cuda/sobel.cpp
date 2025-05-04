#include <torch/extension.h>

void sobel_cuda(torch::Tensor input, torch::Tensor output);

void sobel(torch::Tensor input, torch::Tensor output) {
    sobel_cuda(input, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sobel", &sobel, "Sobel filter wrapper");
}
