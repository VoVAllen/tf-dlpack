/*!
 *  Copyright (c) 2019 by Contributors
 * \file util.h
 * \brief TF data type utilities
 */
#ifndef TF_DLPACK_UTIL_H_
#define TF_DLPACK_UTIL_H_

#include <dlpack/dlpack.h>
#include <tensorflow/core/framework/op_kernel.h>

/*!
 * \brief Convert a DLPack data type object to tensorflow DataType object.
 * \param dtype DLPack dtype object
 * \return tensorflow dtype object
 */
tensorflow::DataType ToTFDataType(const DLDataType &dtype);

#endif  // TF_DLPACK_UTIL_H_
