import tensorflow as tf

FLAGS = None

# Config to turn on JIT compilation
config = tf.ConfigProto(log_device_placement=True)
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.Session(config=config) as sess:
    a = tf.placeholder(tf.float32, [2], name="a")
    b = tf.placeholder(tf.float32, [2], name="b")

    with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
        c = tf.add(a, b, name="c")

    print(sess.run(
        c,
        {
            a: [1.0, 3.0],
            b: [2.0, 5.0],
        }
    ))
