import tensorflow as tf

dirs = {
    'bazel-bin': '/home/jfcherng/Desktop/repo/tensorflow/bazel-bin',
    'user_ops': '/tensorflow/core/user_ops',
}

zero_out = tf.load_op_library(
    dirs['bazel-bin'] + dirs['user_ops'] + '/zero_out.so'
).zero_out

with tf.Session():
    print(
        zero_out([[1, 2], [3, 4]]).eval()
    )
