import argparse
import sys
import tensorflow as tf

FLAGS = None

config = tf.ConfigProto(log_device_placement=True)

# turn on JIT compilation?
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


def main(_):

    device_string = '/job:localhost/replica:0/task:0/device:%s:0' % FLAGS.device

    print('[device string] ' + device_string)

    with tf.Session(config=config) as sess:

        a = tf.constant(3.0, name="a")
        b = tf.placeholder(tf.float32, [], name="b")

        with tf.device(device_string):
            c = tf.add(a, b, name="c")

            result = c

            print(sess.run(
                result,
                {
                    b: 2.0,
                }
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        '-d', '--device', '--device_string',
        dest='device',
        action='store',
        default='CPU',
        help='The TensorFlow device for running this script. (default: CPU)',
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
