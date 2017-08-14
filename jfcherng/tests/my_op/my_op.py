import tensorflow as tf

dirs = {
    'bazel-bin': '/home/jfcherng/Desktop/repo/tensorflow/bazel-bin',
    'user_ops': '/tensorflow/core/user_ops',
}

my_op = tf.load_op_library(
    dirs['bazel-bin'] + dirs['user_ops'] + '/my_op.so'
).my_op


# model a

# 2x2
A_W1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], name="A_W1")
# 2x2
A_x1 = tf.placeholder(tf.float32, [2, 2], name="A_x1")
# 2x2
A_B1 = tf.constant([[5.0, 6.0], [7.0, 8.0]], name="A_B1")

A_MatMul1 = tf.matmul(A_x1, A_W1, name="A_MatMul1")
A_Add1 = tf.matmul(A_MatMul1, A_B1, name="A_Add1")


# 2x2
A_W2 = tf.constant([[-1.0, 2.0], [1.0, -2.0]], name="A_W2")
# 2x2
A_B2 = tf.constant([[1.0, -5.0], [1.0, -3.0]], name="A_B2")
A_MatMul2 = tf.matmul(A_W2, A_Add1, name="A_MatMul2")
A_Add2 = tf.matmul(A_MatMul2, A_B2, name="A_Add2")

A_result = A_Add2

# model B

# 2x2
B_W1 = tf.constant([[1.0, 2.0], [1.0, 2.0]], name="B_W1")
# 2x2
B_x1 = tf.placeholder(tf.float32, [2, 2], name="B_x1")
# 2x2
B_B1 = tf.constant([[1.0, 2.0], [1.0, 2.0]], name="B_B1")

B_MatMul1 = tf.matmul(B_W1, B_x1, name="B_MatMul1")
B_Add1 = tf.matmul(B_MatMul1, B_B1, name="B_Add1")

B_result = B_Add1

# my op

my_op_result = my_op(A_result, B_result)


# GO!
with tf.Session() as sess:

    result = sess.run(
        A_Add1,
        feed_dict = {
            A_x1: [
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        }
    )

    print(result)
