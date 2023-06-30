import time
import unittest
from unittest import TestCase

import torch

from raymarching import near_far_from_aabb, get_backend


class TestNearFarFromAABB(TestCase):

    def setUp(self) -> None:
        super().setUp()

        num_samples = 100
        self.iters = 1000
        dtype = torch.float32
        self.rays_o = torch.randn(num_samples, 3, device="cuda", dtype=dtype)
        rays_d = torch.randn(num_samples, 3, device="cuda", dtype=dtype)
        self.rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        self.aabb = torch.tensor([-1, -1, -1, 1, 1, 1], device="cuda", dtype=dtype)

    def test_aabb(self):
        timer1, timer2 = 0, 0

        for _ in range(self.iters):
            now = time.time()
            nears, fars = near_far_from_aabb(self.rays_o, self.rays_d, self.aabb, 0.2)
            torch.cuda.synchronize()
            timer1 += time.time() - now

        for _ in range(self.iters):
            now = time.time()
            nears, fars = get_backend().near_far_from_aabb2(self.rays_o, self.rays_d, self.aabb, 0.2)
            torch.cuda.synchronize()
            timer2 += time.time() - now

        print("method1", f"{timer1 / self.iters:-1e}")
        print("method2", f"{timer2 / self.iters:-1e}")

    def test_equal(self):

        nears, fars = near_far_from_aabb(self.rays_o, self.rays_d, self.aabb, 0.2)
        nears2, fars2 = get_backend().near_far_from_aabb2(self.rays_o, self.rays_d, self.aabb, 0.2)
        assert torch.allclose(nears, nears2)
        assert torch.allclose(fars, fars2)


if __name__ == '__main__':
    unittest.main()
