/*!
 * Copyright (c) 2019 by Contributors
 * \file util.cc
 * \brief Utility functions
 */
#include "./util.h"

using namespace tensorflow;

DataType ToTFDataType(const DLDataType &dtype) {
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


