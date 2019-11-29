/*!
 * Copyright (c) 2019 by Contributors
 * \file get_device_and_dtype_kernel.cc
 * \brief get device and dtype kernel
 */
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <cstdint>
#include "./util.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class GetDeviceAndDTypeOP : public OpKernel {
 public:
  explicit GetDeviceAndDTypeOP(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    uint64 address = input_tensor.flat<uint64>()(0);
    DLManagedTensor *dl_tensor = static_cast<DLManagedTensor *>(reinterpret_cast<void *>(address));
    Tensor *output_tensor = NULL;
    TensorShape shape = TensorShape({3});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    auto output_flat = output_tensor->flat<int32>();
    output_flat(0) = dl_tensor->dl_tensor.ctx.device_type;
    output_flat(1) = dl_tensor->dl_tensor.ctx.device_id;
    output_flat(2) = ToTFDataType(dl_tensor->dl_tensor.dtype);
  }
};

REGISTER_KERNEL_BUILDER(Name("GetDeviceAndDtype").Device(DEVICE_CPU), GetDeviceAndDTypeOP);
