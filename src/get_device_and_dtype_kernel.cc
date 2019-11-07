/*!
 *  Copyright (c) 2019 by Contributors
 * \file get_device_and_dtype_kernel.cc
 * \brief get device and dtype kernel
 */
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <cstdint>
#include "util.h"

using namespace tensorflow;
namespace tf = tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

DataType toTFDataType(const DLDataType &dtype) {
  DataType tf_dtype = DT_INVALID;
  int code = dtype.code;
  int bits = dtype.bits;
  switch (code) {
    case kDLUInt:
      switch (bits) {
        case 8:
          tf_dtype = DT_UINT8;
          break;
        case 16:
          tf_dtype = DT_UINT16;
          break;
        case 32:
          tf_dtype = DT_UINT32;
          break;
        case 64:
          tf_dtype = DT_UINT64;
          break;
        default:
          LOG(INFO) << "Unsupported kUInt bits";
      }
      break;
    case kDLInt:
      switch (bits) {
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
          LOG(INFO) << "Unsupported kInt bits";
      }
      break;
    case kDLFloat:
      switch (bits) {
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
          LOG(INFO) << "Unsupported kFloat bits";
      }
      break;
    default:
      LOG(INFO) << "Unsupported code";
  }
  return tf_dtype;
}

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
    output_flat(2) = toTFDataType(dl_tensor->dl_tensor.dtype);
  }
};

REGISTER_KERNEL_BUILDER(Name("GetDeviceAndDtype").Device(DEVICE_CPU), GetDeviceAndDTypeOP);
