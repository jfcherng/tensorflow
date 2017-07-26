import tensorflow as tf

FLAGS = None

# Config to turn on JIT compilation
config = tf.ConfigProto(log_device_placement=True)
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.Session(config=config) as sess:

    a = tf.constant([[1.0, -3.0]], name="a")
    b = tf.Variable([[2.0, 5.0, 7.0], [2.0, 5.0, 7.0], ], name="b")
    e = tf.placeholder(tf.float32, [1, 3], name="e")

    sess.run(tf.global_variables_initializer())
    with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
        c = tf.matmul(a, b, name="c")
        result = tf.concat([c, e], 0, name='result')


        reluTest= tf.nn.relu(e, name='reluTest')

    print(sess.run(
        reluTest,
        {
            # a: [
            #     [1.0, 3.0]
            # ],
            # b: [
            #     [2.0, 5.0, 7.0],
            #     [2.0, 5.0, 7.0],
            # ],
            e: [
                [4.0, -5.0, 6.0],
            ],
        }
    ))
