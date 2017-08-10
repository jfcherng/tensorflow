#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include <functional> // std::multiplies
#include <numeric> // std::accumulate

namespace tensorflow {

void relu_op_jfcherng_xla_impl(void *out, void **data) {

  // the input tensor (but represented in a 1-D array)
  float *input = static_cast<float *>(data[0]);
  // the output tensor
  float *output = static_cast<float *>(out);

  // if we have a 3-D, 5 by 3 by 7 tensor, then
  // *input_dim_sizes = [5, 3, 7] and
  // input_dim = 3.
  int64 *input_dim_sizes = static_cast<int64 *>(data[1]);
  int64 input_dim = *static_cast<int64 *>(data[2]);

  // calculate tolal element counts of the input
  int64 elementCounts = std::accumulate(
                          input_dim_sizes,
                          input_dim_sizes + input_dim,
                          1,
                          std::multiplies<int64>()
                        );

  // do ReLU on each element of the input
  for (int i = 0; i < elementCounts; ++i) {
    output[i] = (input[i] > 0.0) ? input[i] : 0.0;
  }

}

}  // namespace tensorflow

// Implements argmax on CPU.
// This is called by an XLA custom call, set up by relu_ops.cc.
extern "C" void TF_EXPORT
relu_op_jfcherng_xla_impl(void* out, void** data) {
  tensorflow::relu_op_jfcherng_xla_impl(out, data);
}
