import tensorflow as tf

# experiment results:
#
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 0 step 1 _SOURCE = NoOp[]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 2 step 1 Model_A/a = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [2] values: 1 2>, _device="/job:localhost/replica:0/task:0/cpu:0"]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 3 step 1 Model_A/b = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [2] values: 2 3>, _device="/job:localhost/replica:0/task:0/cpu:0"]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 5 step 1 Model_B/p = Const[dtype=DT_INT32, value=Tensor<type: int32 shape: [] values: 8>, _device="/job:localhost/replica:0/task:0/cpu:0"]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 6 step 1 Model_B/q = Const[dtype=DT_INT32, value=Tensor<type: int32 shape: [] values: 3>, _device="/job:localhost/replica:0/task:0/cpu:0"]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 7 step 1 Model_B/r.mul = Mul[T=DT_INT32, _device="/job:localhost/replica:0/task:0/cpu:0"](Model_B/p, Model_B/q) is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 4 step 1 Model_A/c.add = Add[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"](Model_A/a, Model_A/b) is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 8 step 1 M.grp = NoOp[_device="/job:localhost/replica:0/task:0/cpu:0"](^Model_A/c.add, ^Model_B/r.mul) is dead: 0
#
# execution order: x -> W -> node_mul -> B -> node_add

with tf.name_scope('Model_A'):
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    c = tf.add(a, b, name="c.add")

with tf.name_scope('Model_B'):
    p = tf.constant(8, name="p")
    q = tf.constant(3, name="q")
    r = tf.multiply(p, q, name="r.mul")

M = tf.group(c, r, name='M.grp')

with tf.Session() as sess:
    sess.run(M)
