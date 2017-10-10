import tensorflow as tf

with tf.name_scope('Model_A'):
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    c = tf.add(a, b, name="c.add")

with tf.name_scope('Model_B'):
    p = tf.constant(8, name="p")
    q = tf.constant(3, name="q")
    r = tf.multiply(p, q, name="r.mul")

with tf.Session() as sess:
    result = sess.run([c, r])

print(result)
