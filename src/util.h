/*!
 *  Copyright (c) 2019 by Contributors
 * \file util.h
 * \brief TF data type utilities
 */

#pragma once

#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

DataType toTFDataType(const DLDataType &dtype);
