
#include <torch/extension.h>
#include "include/bindings.h"
#include <iostream>


int main() {
    at::Tensor ray_o = torch::rand({10, 3});
    at::Tensor ray_d = torch::rand({10, 3});
    at::Tensor nears = torch::empty({10, 3});
    at::Tensor fars = torch::empty({10, 3});
    at::Tensor aabb = torch::tensor({-1, -1, -1, 1, 1, 1});
    near_far_from_aabb(ray_o, ray_d, aabb, 10, 0.5, nears, fars);
    std::cout << nears << std::endl;
    return 0;
}

