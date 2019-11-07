#pragma once

#include <cstdio>
#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

DataType toTFDataType(const DLDataType &dtype);
