#include <cstdio>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/tensor_reference.h>
// #include "tensorflow/core/kernels/tensor_cord.h"

//This file is not included in build

using namespace tensorflow;
namespace tf = tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class DLPackAllocator : public Allocator {
 public:
  static constexpr size_t kAllocatorAlignment = 1;

  DLPackAllocator(DLManagedTensor *dlm_tensor) {
    dlm_tensor_ = dlm_tensor;
    data_ = dlm_tensor->dl_tensor.data;

    // Shape
    shape_ = TensorShape();
    int ndim = dlm_tensor->dl_tensor.ndim;
    int64_t *shape = dlm_tensor->dl_tensor.shape;
    for (int i = 0; i < ndim; i++)
    {
      shape_.AddDim(shape[i]);
    }
    num_elements_ = shape_.num_elements();
  }

  string Name() { return "DLPackAllocator"; }

  void *AllocateRaw(size_t alignment, size_t num_bytes) {
    if (num_elements_ * (dlm_tensor_->dl_tensor.dtype.bits) / 8 != num_bytes)
    {
      std::cout << "Invalid allocation bytes" << std::endl;
    };
    auto iptr = reinterpret_cast<std::uintptr_t>(data_);
    if (!(iptr % alignment))
    {
      std::cout << "Memory not aligned" << std::endl;
    }
    return data_;
  };

  void DeallocateRaw(void *ptr) {
    // This would lead to double free, haven't figure out the problem
    // dlm_tensor_->deleter(const_cast<DLManagedTensor *>(dlm_tensor_));
    std::cout << "Deconstruct dlpack tensor" << std::endl;
    std::cout << (long)ptr << std::endl;
    std::cout << (long)data_ << std::endl;
    // delete this;
  };

  TensorShape get_shape() {
    return shape_;
  }

 private:
  DLManagedTensor *dlm_tensor_;
  void *data_;
  int64 num_elements_;
  TensorShape shape_;
};

DataType toTFDataType(DLDataType dtype) {
  DataType tf_dtype = DT_INVALID;
  switch (dtype.code) {
    case kDLUInt:
      switch (dtype.bits) {
        case 32:
          tf_dtype = DT_UINT32;
          break;
        case 64:
          tf_dtype = DT_UINT64;
          break;
        default:
          0;
          // OP_REQUIRES(context, false, errors::Unimplemented("Unsupported kUInt bits"));
      }
      break;
    case kDLInt:
      switch (dtype.bits) {
        case 8:
          tf_dtype = DT_INT8;
          break;
        case 16:
          tf_dtype = DT_INT16;
          break;
        case 32:
          tf_dtype = DT_INT32;
          break;
        case 64:
          tf_dtype = DT_INT64;
          break;
        default:
          0;
          // OP_REQUIRES(context, false, errors::Unimplemented("Unsupported kInt bits"));
      }
      break;

    case kDLFloat:
      switch (dtype.bits) {
        case 16:
          tf_dtype = DT_HALF;
          break;
        case 32:
          tf_dtype = DT_FLOAT;
          break;
        case 64:
          tf_dtype = DT_DOUBLE;
          break;
        default:
          0;
          // OP_REQUIRES(context, false, errors::Unimplemented("Unsupported kFloat bits"));
      }
      break;
    default:
      0;
      // OP_REQUIRES(context, false, errors::Unimplemented("Unsupported code"));
  }
  return tf_dtype;
}

// template <typename DEVICE_TYPE, typename T>
class FromDLPackOP : public OpKernel {

 public:
  explicit FromDLPackOP(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    uint64 address = input_tensor.flat<uint64>()(0);
    DLManagedTensor *dlm_tensor = static_cast<DLManagedTensor *>((void *)address);
    DLPackAllocator *dlpack_allocator = new DLPackAllocator(dlm_tensor);
    DLDataType dtype = dlm_tensor->dl_tensor.dtype;

    DataType tf_dtype = toTFDataType(dtype);
    Tensor *output_tensor = new Tensor(dlpack_allocator, tf_dtype, dlpack_allocator->get_shape());
    // const Tensor output_tensor(dlpack_allocator, tf_dtype, dlpack_allocator->get_shape());

    context->set_output_ref(0, &mu_, output_tensor);

    output_tensor->tensor_data().data()
    // context->set_output(0, output_tensor);
  }

 private:
  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_CPU), FromDLPackOP);
REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_GPU), FromDLPackOP);
