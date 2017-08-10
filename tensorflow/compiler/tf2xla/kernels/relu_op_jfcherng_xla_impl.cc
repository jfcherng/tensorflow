#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include <iostream>

namespace tensorflow {

void relu_op_jfcherng_xla_impl(void *out, void **data) {

  // data is managed by the JIT code so msan can't tell it's initialized.
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(data, sizeof(void*));

  float *input = static_cast<float *>(data[0]);

  int64 *input_dim_sizes = static_cast<int64 *>(data[1]); // [5, 3]

  int64 *input_dim = static_cast<int64 *>(data[2]); // 2

  // calculate tolal elements from the input
  int64 totalElementCounts = 1;
  for (int i = 0; i < *input_dim; ++i) {
    totalElementCounts *= input_dim_sizes[i];
  }

  float *out_real = static_cast<float *>(out);

  for (int i = 0; i < totalElementCounts; ++i) {
    if (input[i] > 0.0) {
      out_real[i] = input[i];
    } else {
      out_real[i] = 0.0;
    }
  }

}

}  // namespace tensorflow

// Implements argmax on CPU.
// This is called by an XLA custom call, set up by relu_ops.cc.
extern "C" void TF_EXPORT
relu_op_jfcherng_xla_impl(void* out, void** data) {
  tensorflow::relu_op_jfcherng_xla_impl(out, data);
}
