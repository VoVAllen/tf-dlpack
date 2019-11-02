#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ToDLPack")
    .Attr("T: {float, double, int32, int64, uint32, uint64}")
    .Input("tf_tensor : T")
    .Output("add : uint64");