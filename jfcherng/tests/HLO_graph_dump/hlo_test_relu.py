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

        relu_test_M = tf.placeholder(tf.float32, [2, 3], name="M")

        with tf.device(device_string):

            # result = relu(M)
            relu_test_result = tf.nn.relu(relu_test_M)

            print(sess.run(
                relu_test_result,
                {
                    relu_test_M: [
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
