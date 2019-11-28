/*!
 *  Copyright (c) 2019 by Contributors
 * \file util.h
 * \brief TF data type utilities
 */
#ifndef TFDLPACK_UTIL_H_
#define TFDLPACK_UTIL_H_

#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>

/*!
 * \brief Convert a DLPack data type object to tensorflow DataType object.
 * \param dtype DLPack dtype object
 * \return tensorflow dtype object
 */
tensorflow::DataType ToTFDataType(const DLDataType &dtype);

#endif  // TFDLPACK_UTIL_H_
