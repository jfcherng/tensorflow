import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    c = tf.add(a, b, name="c")

    print(sess.run(c))
