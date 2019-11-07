#include <cstdio>
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_reference.h>

using namespace tensorflow;
namespace tf = tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename DEVICE_TYPE>
class DeviceOpTrait;

template <>
class DeviceOpTrait<CPUDevice> {
 public:
  static const DLDeviceType device_type = kDLCPU;
  static int device_id(OpKernelContext *context) {
    return 0;
  }
};

template <>
class DeviceOpTrait<GPUDevice> {
 public:
  static const DLDeviceType device_type = kDLGPU;
  static int device_id(OpKernelContext *context) {
    auto device_base = context->device();
    auto gpu_device_info = device_base->tensorflow_gpu_device_info();
    return gpu_device_info->gpu_id;
  }
};

template <typename DATA_TYPE>
class DataTypeTrait;

#define DATA_TYPE_DISPATCH(T, data_code, data_bits, data_lanes) \
  template <>                                                   \
  class DataTypeTrait<T> {                                      \
   public:                                                      \
    static const uint8_t code = data_code;                      \
    static const uint8_t bits = data_bits;                      \
    static const uint16_t lanes = data_lanes;                   \
  };

DATA_TYPE_DISPATCH(float, kDLFloat, 32, 1);
DATA_TYPE_DISPATCH(double, kDLFloat, 64, 1);
DATA_TYPE_DISPATCH(int32, kDLInt, 32, 1);
DATA_TYPE_DISPATCH(int64, kDLInt, 64, 1);
DATA_TYPE_DISPATCH(uint32, kDLUInt, 32, 1);
DATA_TYPE_DISPATCH(uint64, kDLUInt, 64, 1);

struct TFDLMTensor {
  TensorReference *handle;
  DLManagedTensor tensor;
};

void deleter(DLManagedTensor *arg) {
  TFDLMTensor *owner = static_cast<TFDLMTensor *>(arg->manager_ctx);
  owner->handle->Unref();
  delete owner;
};

template <typename DEVICE_TYPE, typename T>
class ToDLPackOP : public OpKernel {

 public:
  explicit ToDLPackOP(OpKernelConstruction *context) : OpKernel(context) { }
  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &input_tensor = context->input(0);
    DLDeviceType device_type = DeviceOpTrait<DEVICE_TYPE>::device_type;
    int device_id = DeviceOpTrait<DEVICE_TYPE>::device_id(context);
    TFDLMTensor *tfDLMTensor(new TFDLMTensor);
    DLContext ctx = {device_type, device_id};

    DLDataType data_type = {DataTypeTrait<T>::code, DataTypeTrait<T>::bits, DataTypeTrait<T>::lanes};

    TensorReference *tensor_ref = new TensorReference(input_tensor); // This will call buf_->Ref()
    tfDLMTensor->handle = tensor_ref;
    tfDLMTensor->tensor.manager_ctx = tfDLMTensor;
    tfDLMTensor->tensor.deleter = &deleter;
    tfDLMTensor->tensor.dl_tensor.ctx = ctx;
    int ndim = input_tensor.dims();
    tfDLMTensor->tensor.dl_tensor.ndim = ndim;
    tfDLMTensor->tensor.dl_tensor.data = const_cast<void *>((const void *)input_tensor.tensor_data().data());

    tfDLMTensor->tensor.dl_tensor.dtype = data_type;
    input_tensor.shape();
    int64_t *shape_arr = new int64_t[ndim];
    for (int i = 0; i < ndim; i++) {
      shape_arr[i] = input_tensor.dim_size(i);
    }
    tfDLMTensor->tensor.dl_tensor.shape = shape_arr;
    tfDLMTensor->tensor.dl_tensor.strides = nullptr;
    tfDLMTensor->tensor.dl_tensor.byte_offset = 0;
    uint64 add = (uint64)(void *)(&tfDLMTensor->tensor);

    Tensor *output_tensor = NULL;
    TensorShape shape = TensorShape({1});
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor, attr));
    auto output_flat = output_tensor->flat<uint64>();
    output_flat(0) = add;
  }
};

#define REGISTER_KERNEL_DISPATCH(T) \
  REGISTER_KERNEL_BUILDER(Name("ToDlpack").Device(DEVICE_CPU).TypeConstraint<T>("T"), ToDLPackOP<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ToDlpack").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("out"), ToDLPackOP<GPUDevice, T>);

REGISTER_KERNEL_DISPATCH(float);
REGISTER_KERNEL_DISPATCH(double);
REGISTER_KERNEL_DISPATCH(int32);
REGISTER_KERNEL_DISPATCH(int64);
REGISTER_KERNEL_DISPATCH(uint32);
REGISTER_KERNEL_DISPATCH(uint64);

#undef REGISTER_KERNEL_DISPATCH
