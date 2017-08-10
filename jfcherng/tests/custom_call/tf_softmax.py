import argparse
import sys
import tensorflow as tf
import numpy as np

FLAGS = None

config = tf.ConfigProto(log_device_placement=True)

# turn on JIT compilation?
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# scalar test
all_test_cases = [

    # 1-D test cases
    [
        # 1-D: 7
        [1, 2, 3, 4, 1, 2, 3],
    ],

    # 2-D test cases
    [
        # 2-D: 5x3
        [
            [12.34, -56.78, 0.1],
            [12.34, -56.78, 0.1],
            [12.34, -56.78, 0.1],
            [12.34, -56.78, 0.1],
            [12.34, -56.78, 0.1],
        ],
    ],

    # 3-D test cases
    [
        # 3-D: 2x5x3
        [
            [
                [-2.34, 56.78, -0.1],
                [-2.34, 56.78, -0.1],
                [-2.34, 56.78, -0.1],
                [-2.34, 56.78, -0.1],
                [-2.34, 56.78, -0.1],
            ],
            [
                [12.34, -56.78, 0.1],
                [12.34, -56.78, 0.1],
                [12.34, -56.78, 0.1],
                [12.34, -56.78, 0.1],
                [12.34, -56.78, 0.1],
            ]
        ],
    ]

]


def absDiff(a, b):
    return abs(a-b)


def main(_):

    if FLAGS.xla:
        device_string = '/job:localhost/replica:0/task:0/device:XLA_CPU:0'
    else:
        device_string = '/job:localhost/replica:0/task:0/device:CPU:0'

    for test_cases in all_test_cases:

        with tf.Session(config=config) as sess:
            with tf.device(device_string):

                test_case_shape = np.array(test_cases[0]).shape
                print('[custom_call] shape =', test_case_shape)

                x = tf.placeholder(tf.float32, np.array(test_cases[0]).shape, name="x")
                output = tf.nn.softmax(x, dim=-1, name="output")

                # run test cases
                for test_case in test_cases:

                    result = sess.run(output, { x: test_case })

                    t_shape = np.array(test_case).size
                    test_case = np.array(test_case).reshape((1, t_shape))
                    result = np.array(result).reshape((1, t_shape))

                    print('[custom_call] ', 'input'.rjust(10), ' -> ', 'result'.ljust(10))
                    for (inputs_, outputs_) in (zip(test_case, result)):
                        for (input_, output_) in (zip(inputs_, outputs_)):
                            print('[custom_call] ', (str(input_)).rjust(10), ' -> ', (str(output_)).ljust(10))
                    print('[custom_call]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--xla', dest='xla', action='store_true', help='Turn on XLA. (default)')
    parser.add_argument('--no-xla', dest='xla', action='store_false', help='Turn off XLA.')
    parser.set_defaults(xla=True)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
