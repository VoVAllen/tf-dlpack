/*!
 * Copyright (c) 2019 by Contributors
 * \file from_dlpack_kernel.cc
 * \brief from dlpack kernel
 */
#ifdef TFDLPACK_USE_CUDA
#include <cuda_runtime.h>
#endif  // TFDLPACK_USE_CUDA
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_reference.h>
#include <cstdio>
#include "./util.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

inline bool IsAligned(size_t alignment, void *data_ptr) {
  auto iptr = reinterpret_cast<std::uintptr_t>(data_ptr);
  return (iptr % alignment == 0);
}

class DLPackAllocator : public Allocator {
 public:
  explicit DLPackAllocator(DLManagedTensor *dlm_tensor) {
    dlm_tensor_ = dlm_tensor;
    data_ = dlm_tensor->dl_tensor.data;

    // Shape
    shape_ = TensorShape();
    const int ndim = dlm_tensor->dl_tensor.ndim;
    const int64_t *shape = dlm_tensor->dl_tensor.shape;
    for (int i = 0; i < ndim; i++) {
      shape_.AddDim(shape[i]);
    }
    num_elements_ = shape_.num_elements();
  }

  string Name() { return "DLPackAllocator"; }

  void *AllocateRaw(size_t alignment, size_t num_bytes) {
    if (num_elements_ * (dlm_tensor_->dl_tensor.dtype.bits) / 8 != num_bytes) {
      allocation_status_ =
          errors::Internal("Invalid number of bytes for DLPack Tensor");
      return nullptr;
    }
    if (IsAligned(alignment, data_)) {
      return data_;
    } else {
      allocation_status_ =
          errors::Internal("DLPack Tensor has wrong alignment");
      return nullptr;
    }
  }

  void DeallocateRaw(void *ptr) {
    // This would lead to double free, haven't figure out the problem
    dlm_tensor_->deleter(const_cast<DLManagedTensor *>(dlm_tensor_));
    delete this;
  }

  const Status &allocation_status() const { return allocation_status_; }

  TensorShape get_shape() { return shape_; }
  int64 get_size() {
    return num_elements_ * (dlm_tensor_->dl_tensor.dtype.bits) / 8;
  }

 private:
  DLManagedTensor *dlm_tensor_;
  void *data_;
  int64 num_elements_;
  TensorShape shape_;
  Status allocation_status_;

  TF_DISALLOW_COPY_AND_ASSIGN(DLPackAllocator);
};

class FromDLPackOP : public OpKernel {
 public:
  explicit FromDLPackOP(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    uint64 address = input_tensor.flat<uint64>()(0);
    DLManagedTensor *dlm_tensor =
        static_cast<DLManagedTensor *>(reinterpret_cast<void *>(address));
    DLDataType dtype = dlm_tensor->dl_tensor.dtype;

    DLPackAllocator *dlpack_allocator = new DLPackAllocator(dlm_tensor);
    // Alignment is always 64 bytes for CPU and GPU in TF
    if (IsAligned(64, dlm_tensor->dl_tensor.data)) {
      // Aligned tensor using DLPackAllocator to allocate memory
      DataType tf_dtype = ToTFDataType(dtype);
      Tensor output_tensor(dlpack_allocator, tf_dtype,
                           dlpack_allocator->get_shape());
      OP_REQUIRES_OK(context, dlpack_allocator->allocation_status());
      OP_REQUIRES_OK(context, context->set_output("out", output_tensor));
    } else {
      // Copy unaligned tensor and using tf allocator
      Tensor *output_tensor;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, dlpack_allocator->get_shape(),
                                              &output_tensor));
      void *tftensor_ptr =
          const_cast<void *>((const void *)output_tensor->tensor_data().data());
      void *dlpack_ptr = dlm_tensor->dl_tensor.data;
      size_t size = dlpack_allocator->get_size();
      if (dlm_tensor->dl_tensor.ctx.device_type == kDLCPU) {
        memcpy(tftensor_ptr, dlpack_ptr, size);
      } else if (dlm_tensor->dl_tensor.ctx.device_type == kDLGPU) {
#ifdef TFDLPACK_USE_CUDA
        cudaMemcpy(tftensor_ptr, dlpack_ptr, size, cudaMemcpyDeviceToDevice);
#endif  // TFDLPACK_USE_CUDA
      } else {
        OP_REQUIRES_OK(context, errors::Internal("Device unsupported"));
      }
      dlpack_allocator->DeallocateRaw(nullptr);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_CPU), FromDLPackOP);
REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_GPU).HostMemory("in"),
                        FromDLPackOP);
