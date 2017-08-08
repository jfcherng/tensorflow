import argparse
import sys
import tensorflow as tf

FLAGS = None

config = tf.ConfigProto(log_device_placement=True)

# turn on JIT compilation?
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

test_op_name = "relu"

# scalar test
test_cases = [ 12.34, -56.78, 0.0 ]
test_cases_2 = [ [12.34, -56.78] ]


def main(_):

    if FLAGS.xla:
        device_string = '/job:localhost/replica:0/task:0/device:XLA_CPU:0'
    else:
        device_string = '/job:localhost/replica:0/task:0/device:CPU:0'

    with tf.Session(config=config) as sess:
        with tf.device(device_string):

            t = test_cases
            x = tf.placeholder(tf.float32, [], name="x")

            output = tf.nn.relu(x, name="output")
            # run test cases
            for test_case in t:
                result = sess.run(output, { x: test_case })
                print(
                    "[custom_call]\n"
                    "\tx = {}\n"
                    "\t{}(x) = {}".format(
                        test_case, test_op_name, result
                    )
                )

    with tf.Session(config=config) as sess:
        with tf.device(device_string):

            t = test_cases_2
            x = tf.placeholder(tf.float32, [2], name="x")

            output = tf.nn.relu(x, name="output")
            # run test cases
            for test_case in t:
                result = sess.run(output, { x: test_case })
                print(
                    "[custom_call]\n"
                    "\tx = {}\n"
                    "\t{}(x) = {}".format(
                        test_case, test_op_name, result
                    )
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--xla', dest='xla', action='store_true', help='Turn on XLA. (default)')
    parser.add_argument('--no-xla', dest='xla', action='store_false', help='Turn off XLA.')
    parser.set_defaults(xla=True)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
