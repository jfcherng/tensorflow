#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include <cmath>

namespace tensorflow {

void softmax_op_hsien_xla_impl(void *out, void **data) {
  // data is managed by the JIT code so msan can't tell it's initialized.
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(data, sizeof(void*));

  // the input tensor (an 2-D matrix but in its 1-D representation)
  float *input = static_cast<float *>(data[0]);
  // the output tensor
  float *output = static_cast<float *>(out);

  // if we have a 3-D, 5 by 3 by 7 tensor, then
  // *input_dim_sizes = [5, 3, 7] and
  // input_dim = 3.
  int64 *input_dim_sizes = static_cast<int64 *>(data[1]);
  int64 input_dim = *static_cast<int64 *>(data[2]);

  // calculate tolal elements from the input
  int64 totalElementCounts = 1;
  for (int i = 0; i < input_dim; ++i) {
    totalElementCounts *= input_dim_sizes[i];
  }

  for (int i = 0; i < totalElementCounts; ++i) {
    input[i] = std::exp(input[i]);
  }

  float rowSum = 0;
  for (int i = 0; i < input_dim_sizes[0]; ++i) {
    rowSum = 0;
    for (int j = 0; j < input_dim_sizes[1]; ++j) {
      rowSum += input[i * input_dim_sizes[1] + j];
    }
    for (int k = 0; k < input_dim_sizes[1]; ++k) {
      output[i * input_dim_sizes[1] + k] = input[i * input_dim_sizes[1] + k] / rowSum;
    }
  }

}

}  // namespace tensorflow

// Implements argmax on CPU.
// This is called by an XLA custom call, set up by softmax_ops.cc.
extern "C" void TF_EXPORT
softmax_op_hsien_xla_impl(void* out, void** data) {
  tensorflow::softmax_op_hsien_xla_impl(out, data);
}
