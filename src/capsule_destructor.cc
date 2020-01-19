/*!
 * Copyright (c) 2019 by Contributors
 * \file capsule_destructor.cc
 * \brief Call destructor of DLManagedTensor
 */
#ifdef TFDLPACK_USE_CUDA
#include <cuda_runtime.h>
#endif  // TFDLPACK_USE_CUDA
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_reference.h>
#include <cstdio>
#include "./util.h"

using namespace tensorflow;

class DestructOP : public OpKernel {
 public:
  explicit DestructOP(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    uint64 address = input_tensor.flat<uint64>()(0);
    DLManagedTensor *dlm_tensor =
        static_cast<DLManagedTensor *>(reinterpret_cast<void *>(address));
    dlm_tensor->deleter(const_cast<DLManagedTensor *>(dlm_tensor));
  }
};

REGISTER_KERNEL_BUILDER(Name("DestructDLPACK").Device(DEVICE_CPU), DestructOP);