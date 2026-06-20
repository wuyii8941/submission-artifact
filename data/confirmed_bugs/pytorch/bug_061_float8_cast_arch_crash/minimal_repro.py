import torch

@torch.compile(backend="inductor")
def f(x):
    return x.to(torch.float8_e4m3fn)

x = torch.randn(16, device="cuda")
eager_out = x.to(torch.float8_e4m3fn)
compiled_out = f(x)  # crashes on GPUs with compute capability < 8.9
