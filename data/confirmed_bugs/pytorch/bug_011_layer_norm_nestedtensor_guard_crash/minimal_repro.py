import torch
import torch.nn.functional as F

offsets = torch.tensor([0, 3, 7, 12])
values = torch.randn(12, 16, requires_grad=True)
nt = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)

# Eager: works
out_eager = F.layer_norm(nt, [16])
print(f"eager ok: shape={out_eager.shape}")

# Compiled: crashes in guard builder
torch._dynamo.reset()
compiled_ln = torch.compile(lambda x: F.layer_norm(x, [16]))
try:
    out_compiled = compiled_ln(nt)
    print(f"compiled ok: shape={out_compiled.shape}")
except Exception as e:
    print(f"compiled CRASHED: {type(e).__name__}: {e}")
