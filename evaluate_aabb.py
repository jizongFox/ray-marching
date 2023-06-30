import time

import torch

from raymarching import near_far_from_aabb, get_backend

num_samples = 1
iters = 1000
method1 = 0
method2 = 0
dtype = torch.float16
rays_o = torch.randn(num_samples, 3, device="cuda", dtype=dtype)
rays_d = torch.randn(num_samples, 3, device="cuda", dtype=dtype)
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
aabb = torch.tensor([-1, -1, -1, 1, 1, 1], device="cuda", dtype=dtype)

for _ in range(iters):
    now = time.time()
    nears, fars = near_far_from_aabb(rays_o, rays_d, aabb)
    torch.cuda.synchronize()
    method1 += time.time() - now

for _ in range(iters):
    now = time.time()
    nears, fars = get_backend().near_far_from_aabb2(rays_o, rays_d, aabb, 0.2)
    torch.cuda.synchronize()
    method2 += time.time() - now

print("method1", f"{method1 / iters:-1e}")
print("method2", f"{method2 / iters:-1e}")

nears, fars = near_far_from_aabb(rays_o, rays_d, aabb)
nears2, fars2 = get_backend().near_far_from_aabb2(rays_o, rays_d, aabb, 0.2)
assert torch.allclose(nears, nears2)
assert torch.allclose(fars, fars2)
