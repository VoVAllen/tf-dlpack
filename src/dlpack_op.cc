#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// Stupid Tensorflow will turn the camelcase name into snakecase name
// To make the name be to_dlpack, the OP is regestered as ToDlpack
REGISTER_OP("ToDlpack")
    .Attr("T: {float, double, int32, int64, uint32, uint64}")
    .Input("in : T")
    .Output("out : uint64");