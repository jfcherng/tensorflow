import argparse
import sys
import tensorflow as tf

FLAGS = None

modelFile = 'model/helloworld.meta'

config = tf.ConfigProto(log_device_placement=True)

# turn on JIT compilation?
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


def main(_):

    if FLAGS.xla:
        device_string = '/job:localhost/replica:0/task:0/device:XLA_CPU:0'
    else:
        device_string = '/job:localhost/replica:0/task:0/device:CPU:0'

    saver = tf.train.import_meta_graph(modelFile)

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        with tf.device(device_string):

            result = sess.run(
                "c:0",
                feed_dict = {
                    # 3x5
                    "b:0": [
                        [ 2.0, 7.0, 11.0, -3.0, -7.0 ],
                        [ 2.1, 7.1, 11.1, -3.1, -7.1 ],
                        [ 2.2, 7.2, 11.2, -3.2, -7.2 ],
                    ],
                }
            )

        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--xla', dest='xla', action='store_true', help='Turn on XLA. (default)')
    parser.add_argument('--no-xla', dest='xla', action='store_false', help='Turn off XLA.')
    parser.set_defaults(xla=True)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
