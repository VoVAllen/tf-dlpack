#include <cstdio>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/tensor_reference.h>
// #include "tensorflow/core/kernels/tensor_cord.h"

using namespace tensorflow;
namespace tf = tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename DEVICE_TYPE, typename T>
class FromDLPackOP : public OpKernel {
 public:
  explicit FromDLPackOP(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    uint64 address = input_tensor.flat<uint64>()(0);
    DLManagedTensor *dlm_tensor = static_cast<DLManagedTensor *>((void *)address);
    DLDataType dtype = dlm_tensor->dl_tensor.dtype;
    DLDeviceType device = dlm_tensor->dl_tensor.ctx.device_type;

    // Shape
    TensorShape shape_ = TensorShape();
    int ndim = dlm_tensor->dl_tensor.ndim;
    int64_t *shape = dlm_tensor->dl_tensor.shape;
    for (int i = 0; i < ndim; i++) {
      shape_.AddDim(shape[i]);
    }
    int64 num_elements_ = shape_.num_elements();

    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_, &output_tensor));
    auto output_flat = output_tensor->flat<T>();

    if (device == kDLCPU) {
      memcpy(output_flat.data(),
             dlm_tensor->dl_tensor.data,
             num_elements_ * (dlm_tensor->dl_tensor.dtype.bits) / 8);
    } else {
      cudaMemcpy(output_flat.data(), dlm_tensor->dl_tensor.data,
                 num_elements_ * (dlm_tensor->dl_tensor.dtype.bits) / 8,
                 cudaMemcpyDeviceToDevice);
    }
    dlm_tensor->deleter(const_cast<DLManagedTensor *>(dlm_tensor));
  }

 private:
  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_CPU), FromDLPackOP<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("FromDlpack").Device(DEVICE_GPU), FromDLPackOP<GPUDevice, float>);
