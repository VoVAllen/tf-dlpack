/*!
 * Copyright (c) 2019 by Contributors
 * \file dlpack_op.cc
 * \brief dlpack op registration
 */
#include <tensorflow/core/framework/op.h>

using namespace tensorflow;

// Stupid Tensorflow will turn the camelcase name into snakecase name
// To make the name be to_dlpack, the OP is regestered as ToDlpack
REGISTER_OP("ToDlpack")
  .Attr("T: {half, float, double, int8, int16, int32, int64, uint32, uint64}")
  .Input("in : T")
  .Output("out : uint64");

REGISTER_OP("FromDlpack")
  .Attr("T: {half, float, double, int8, int16, int32, int64, uint32, uint64}")
  .Input("in : uint64")
  .Output("out : T");

// Return a tensor with shape {3}. First is device type, and second is device id.
// Third is the corresponding value of TF_DTYPE
// Device type is based on DLPack Protocal
REGISTER_OP("GetDeviceAndDtype")
  .Input("in : uint64")
  .Output("out : int32");
