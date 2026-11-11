/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Minimal host-tensor stand-in for libtorch. Only the operations
// KVCacheStore uses (shape/dtype/contiguity/block view/data pointer) exist.
// Linked into one CPU-only test target so KVCacheStore's real batching code
// can be compiled and exercised without an NPU/CUDA toolchain.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

namespace torch {

using ScalarType = int32_t;
inline constexpr ScalarType kFloat32 = 0;
inline constexpr int32_t kCPU = 0;

struct TensorOptions final {
  ScalarType dtype_ = kFloat32;
  int32_t device_ = kCPU;

  TensorOptions& dtype(ScalarType dtype) {
    dtype_ = dtype;
    return *this;
  }
  TensorOptions& device(int32_t device) {
    device_ = device;
    return *this;
  }
};

class Tensor final {
 public:
  Tensor() = default;
  Tensor(std::vector<int64_t> shape,
         std::shared_ptr<std::vector<char>> storage,
         size_t offset,
         ScalarType dtype)
      : shape_(std::move(shape)),
        storage_(std::move(storage)),
        offset_(offset),
        dtype_(dtype) {}

  bool defined() const { return storage_ != nullptr; }
  bool is_contiguous() const { return true; }
  int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
  int64_t size(int64_t axis) const { return shape_[static_cast<size_t>(axis)]; }
  ScalarType scalar_type() const { return dtype_; }
  int64_t element_size() const { return sizeof(float); }

  int64_t numel() const {
    if (shape_.empty()) {
      return 0;
    }
    return std::accumulate(shape_.begin(),
                           shape_.end(),
                           static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
  }

  void* data_ptr() const { return storage_->data() + offset_; }

  Tensor operator[](int64_t block) const {
    std::vector<int64_t> block_shape(shape_.begin() + 1, shape_.end());
    const int64_t block_bytes = std::accumulate(block_shape.begin(),
                                                block_shape.end(),
                                                static_cast<int64_t>(1),
                                                std::multiplies<int64_t>()) *
                                element_size();
    return Tensor(
        std::move(block_shape),
        storage_,
        offset_ + static_cast<size_t>(block) * static_cast<size_t>(block_bytes),
        dtype_);
  }

 private:
  std::vector<int64_t> shape_;
  std::shared_ptr<std::vector<char>> storage_;
  size_t offset_ = 0;
  ScalarType dtype_ = kFloat32;
};

inline Tensor zeros(std::vector<int64_t> shape,
                    const TensorOptions& options = {}) {
  int64_t elements = std::accumulate(shape.begin(),
                                     shape.end(),
                                     static_cast<int64_t>(1),
                                     std::multiplies<int64_t>());
  auto storage = std::make_shared<std::vector<char>>(
      static_cast<size_t>(elements) * sizeof(float), '\0');
  return Tensor(std::move(shape), std::move(storage), 0, options.dtype_);
}

}  // namespace torch
