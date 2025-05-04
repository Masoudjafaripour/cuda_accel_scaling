import torch
import torchvision.transforms as T
from PIL import Image
import time
import sobel_cuda

img = Image.open("image.jpg").convert("L")
tensor = T.ToTensor()(img).squeeze(0).contiguous().cuda()
output = torch.zeros_like(tensor)

# Warmup
sobel_cuda.sobel(tensor, output)

# Benchmark
start = time.time()
sobel_cuda.sobel(tensor, output)
torch.cuda.synchronize()
print("CUDA Time:", time.time() - start)

# CPU version for comparison
start = time.time()
sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
sobel_x = sobel_x.view(1, 1, 3, 3).cuda()
sobel_y = sobel_y.view(1, 1, 3, 3).cuda()
img_tensor = tensor.unsqueeze(0).unsqueeze(0)
gx = torch.nn.functional.conv2d(img_tensor, sobel_x, padding=1)
gy = torch.nn.functional.conv2d(img_tensor, sobel_y, padding=1)
magnitude = torch.sqrt(gx ** 2 + gy ** 2)
torch.cuda.synchronize()
print("Torch Conv2D Time:", time.time() - start)
