/*!
 *  Copyright (c) 2019 by Contributors
 * \file from_dlpack_kernel.cc
 * \brief from dlpack kernel
 */
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_reference.h>
#include <cstdio>
#include "util.h"

using namespace tensorflow;
namespace tf = tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class DLPackAllocator : public Allocator {
 public:
  static constexpr size_t kAllocatorAlignment = 1;

  explicit DLPackAllocator(DLManagedTensor *dlm_tensor) {
    dlm_tensor_ = dlm_tensor;
    data_ = dlm_tensor->dl_tensor.data;

    // Shape
    shape_ = TensorShape();
    int ndim = dlm_tensor->dl_tensor.ndim;
    int64_t *shape = dlm_tensor->dl_tensor.shape;
    for (int i = 0; i < ndim; i++) {
      shape_.AddDim(shape[i]);
    }
    num_elements_ = shape_.num_elements();
  }

  string Name() { return "DLPackAllocator"; }

  void *AllocateRaw(size_t alignment, size_t num_bytes) {
    if (num_elements_ * (dlm_tensor_->dl_tensor.dtype.bits) / 8 != num_bytes) {
      std::cout << "Invalid allocation bytes" << std::endl;
    }
    auto iptr = reinterpret_cast<std::uintptr_t>(data_);
    if (!(iptr % alignment)) {
      std::cout << "Memory not aligned" << std::endl;
    }
    return data_;
  }

  void DeallocateRaw(void *ptr) {
    // This would lead to double free, haven't figure out the problem
    dlm_tensor_->deleter(const_cast<DLManagedTensor *>(dlm_tensor_));
    // std::cout << "Deconstruct dlpack tensor" << std::endl;
    delete this;
  }

  TensorShape get_shape() { return shape_; }

 private:
  DLManagedTensor *dlm_tensor_;
  void *data_;
  int64 num_elements_;
  TensorShape shape_;
};

class FromDLPackOP : public OpKernel {
 public:
  explicit FromDLPackOP(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    uint64 address = input_tensor.flat<uint64>()(0);
    DLManagedTensor *dlm_tensor = static_cast<DLManagedTensor *>(reinterpret_cast<void *>(address));
    DLDataType dtype = dlm_tensor->dl_tensor.dtype;

    DLPackAllocator *dlpack_allocator = new DLPackAllocator(dlm_tensor);
    DataType tf_dtype = toTFDataType(dtype);
    Tensor output_tensor(dlpack_allocator, tf_dtype, dlpack_allocator->get_shape());
    OP_REQUIRES_OK(context, context->set_output("out", output_tensor));
  }
};

REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_CPU), FromDLPackOP);
REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_GPU).HostMemory("in"), FromDLPackOP);
