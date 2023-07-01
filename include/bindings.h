#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


void
near_far_from_aabb(at::Tensor rays_o, at::Tensor rays_d, at::Tensor aabb, uint32_t N, float min_near, at::Tensor nears,
                   at::Tensor fars);

std::vector<torch::Tensor>
near_far_from_aabb2(at::Tensor rays_o, at::Tensor rays_d, at::Tensor aabb, float min_near);

void
sph_from_ray(at::Tensor rays_o, at::Tensor rays_d, float radius, uint32_t N, at::Tensor coords);

void morton3D(at::Tensor coords, uint32_t N, at::Tensor indices);

void morton3D_invert(at::Tensor indices, uint32_t N, at::Tensor coords);

void packbits(at::Tensor grid, uint32_t N, float density_thresh, at::Tensor bitfield);

void flatten_rays(at::Tensor rays, uint32_t N, uint32_t M, at::Tensor res);

void march_rays_train(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, float bound,
                      bool contract, float dt_gamma, uint32_t max_steps, uint32_t N,
                      uint32_t C, uint32_t H, at::Tensor nears, at::Tensor fars,
                      at::optional<at::Tensor> xyzs, at::optional<at::Tensor> dirs, at::optional<at::Tensor> ts,
                      at::Tensor rays, at::Tensor counter, at::Tensor noises);

void
composite_rays_train_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor ts, at::Tensor rays,
                             uint32_t M, uint32_t N, float T_thresh, at::Tensor weights,
                             at::Tensor weights_sum, at::Tensor depth, at::Tensor image);

void composite_rays_train_backward(at::Tensor grad_weights, at::Tensor grad_weights_sum,
                                   at::Tensor grad_depth, at::Tensor grad_image, at::Tensor sigmas,
                                   at::Tensor rgbs, at::Tensor ts, at::Tensor rays,
                                   at::Tensor weights_sum, at::Tensor depth, at::Tensor image,
                                   uint32_t M, uint32_t N, float T_thresh, at::Tensor grad_sigmas,
                                   at::Tensor grad_rgbs);

void march_rays(uint32_t n_alive, uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t,
                at::Tensor rays_o, at::Tensor rays_d, float bound, bool contract,
                float dt_gamma, uint32_t max_steps, uint32_t C, uint32_t H,
                at::Tensor grid, at::Tensor nears, at::Tensor fars, at::Tensor xyzs, at::Tensor dirs,
                at::Tensor ts, at::Tensor noises);

void composite_rays(uint32_t n_alive, uint32_t n_step, float T_thresh, at::Tensor rays_alive,
                    at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor ts, at::Tensor weights_sum,
                    at::Tensor depth, at::Tensor image);