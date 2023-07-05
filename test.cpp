
#include <torch/extension.h>
#include "include/bindings.h"
#include <iostream>


int main() {
    at::Tensor ray_o = torch::rand({10, 3}, {torch::kFloat32}).to(torch::kCUDA);
    at::Tensor ray_d = torch::rand({10, 3}, torch::kFloat32).to(torch::kCUDA);
    at::Tensor nears = torch::empty({10, 3}, torch::kFloat32).to(torch::kCUDA);
    at::Tensor fars = torch::empty({10, 3}, torch::kFloat32).to(torch::kCUDA);
    at::Tensor aabb = torch::tensor({-1, -1, -1, 1, 1, 1}, torch::kFloat32).to(torch::kCUDA);
    near_far_from_aabb(ray_o, ray_d, aabb, 10, 0.5, nears, fars);
    std::cout << nears << fars << std::endl;
    at::Tensor near2, far2;
    auto result = near_far_from_aabb2(ray_o.to(torch::kCUDA), ray_d.to(at::kCUDA), aabb.to(at::kCUDA), 0.2);
    near2, far2 = result[0], result[1];
    std::cout << near2 << far2 << std::endl;
    std::cout << (near2 == nears) << std::endl;

    return 0;
}

