#!/usr/bin/env python3
"""
BUG: torch.compile + create_graph=True crashes on variance normalization
due to donated buffer optimization.

Inductor donates backward buffers that create_graph=True still needs,
causing a RuntimeError. Only triggers at specific shape (4, 64) with
the combination of x.mean() and x.var() in the same expression.

Eager mode: OK
Compiled + create_graph=False: OK
Compiled + create_graph=True: CRASH

Workaround: torch._functorch.config.donated_buffer = False
"""
import torch

print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")


def var_norm(x):
    return (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + 1e-5).sqrt()


# 1. Eager: works
print("1. Eager + create_graph=True: ", end="")
torch.manual_seed(42)
x = torch.randn(4, 64, device='cuda', requires_grad=True)
out = var_norm(x).sum()
g = torch.autograd.grad(out, x, create_graph=True)[0]
print(f"OK (grad.requires_grad={g.requires_grad})")

# 2. Compiled + create_graph=False: works
print("2. Compiled + create_graph=False: ", end="")
torch._dynamo.reset()
torch.manual_seed(42)
x2 = torch.randn(4, 64, device='cuda', requires_grad=True)
out2 = torch.compile(var_norm, backend='inductor')(x2).sum()
g2 = torch.autograd.grad(out2, x2, create_graph=False)[0]
print(f"OK (diff vs eager: {(g.detach()-g2).abs().max().item():.2e})")

# 3. Compiled + create_graph=True: CRASHES
print("3. Compiled + create_graph=True: ", end="")
torch._dynamo.reset()
torch.manual_seed(42)
x3 = torch.randn(4, 64, device='cuda', requires_grad=True)
try:
    out3 = torch.compile(var_norm, backend='inductor')(x3).sum()
    g3 = torch.autograd.grad(out3, x3, create_graph=True)[0]
    print(f"OK (unexpected)")
except RuntimeError as e:
    print(f"CRASH!")
    print(f"   Error: {str(e)[:150]}")

# 4. Also crashes with retain_graph=True
print("4. Compiled + retain_graph=True: ", end="")
torch._dynamo.reset()
torch.manual_seed(42)
x4 = torch.randn(4, 64, device='cuda', requires_grad=True)
try:
    out4 = torch.compile(var_norm, backend='inductor')(x4).sum()
    out4.backward(retain_graph=True)
    print(f"OK (unexpected)")
except RuntimeError as e:
    print(f"CRASH!")
    print(f"   Error: {str(e)[:150]}")

# 5. Workaround
print("5. Workaround (donated_buffer=False): ", end="")
torch._dynamo.reset()
import torch._functorch.config
torch._functorch.config.donated_buffer = False
torch.manual_seed(42)
x5 = torch.randn(4, 64, device='cuda', requires_grad=True)
out5 = torch.compile(var_norm, backend='inductor')(x5).sum()
g5 = torch.autograd.grad(out5, x5, create_graph=True)[0]
print(f"OK (diff vs eager: {(g.detach()-g5.detach()).abs().max().item():.2e})")
torch._functorch.config.donated_buffer = True
torch._dynamo.reset()
