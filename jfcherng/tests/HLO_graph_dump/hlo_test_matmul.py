import argparse
import sys
import tensorflow as tf

FLAGS = None

config = tf.ConfigProto(log_device_placement=True)

# turn on JIT compilation?
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


def main(_):

    if FLAGS.xla:
        device_string = '/job:localhost/replica:0/task:0/device:XLA_CPU:0'
    else:
        device_string = '/job:localhost/replica:0/task:0/device:CPU:0'

    with tf.Session(config=config) as sess:

        matmul_test_W = tf.Variable([
            [1.0, -3.0],
        ], name="W")

        matmul_test_B = tf.Variable([
            [2.0, 5.0, 7.0],
            [2.0, 5.0, 7.0],
        ], name="B")

        matmul_test_x = tf.placeholder(tf.float32, [2, 3], name="x")

        sess.run(tf.global_variables_initializer())

        with tf.device(device_string):
            matmul_test_Wx = tf.matmul(matmul_test_W, matmul_test_x, name="Wx")
            matmul_test_WxB = tf.add(matmul_test_Wx, matmul_test_B, name="WxB")

            # result = (W matmul X) add B
            matmul_test_result = matmul_test_WxB

            print(sess.run(
                matmul_test_result,
                {
                    matmul_test_x: [
                        [-2.0, 5.0, -3.0],
                        [1.0, -5.0, 7.0],
                    ],
                }
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--xla', dest='xla', action='store_true', help='Turn on XLA. (default)')
    parser.add_argument('--no-xla', dest='xla', action='store_false', help='Turn off XLA.')
    parser.set_defaults(xla=True)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
