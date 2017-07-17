import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    c = tf.add(a, b, name="c")

with tf.Session() as sess:
    p = tf.constant(8, name="p")
    q = tf.constant(3, name="q")
    r = tf.multiply(p, q, name="r")

with tf.Session() as sess:
    M = tf.group(c, r, name='M')
    sess.run(M)
