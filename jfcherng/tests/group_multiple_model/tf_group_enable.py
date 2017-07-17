import tensorflow as tf

# tensorflow/core/common_runtime/executor.cc:1557] Process node: 0 step -1 _SOURCE = NoOp[]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 2 step -1 x = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [2,1] values: [3][5]>, _device="/job:localhost/replica:0/task:0/cpu:0"]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 3 step -1 W = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [2] values: 1 2>, _device="/job:localhost/replica:0/task:0/cpu:0"]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 4 step -1 node_mul = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"](W, x) is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 5 step -1 B = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [] values: 7>, _device="/job:localhost/replica:0/task:0/cpu:0"](^node_mul) is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 6 step -1 node_add = Add[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"](node_mul, B) is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 7 step -1 _send_node_add_0 = _Send[T=DT_FLOAT, client_terminated=true, recv_device="/cpu:0", send_device="/cpu:0", send_device_incarnation=-7039685446835334635, tensor_name="node_add:0"](node_add) is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 0 step 1 _SOURCE = NoOp[]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 3 step 1 node_add/_0__cf__0 = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [2,2] values: [10 13][12]...>, _device="/job:localhost/replica:0/task:0/cpu:0"]() is dead: 0
# tensorflow/core/common_runtime/executor.cc:1557] Process node: 2 step 1 _retval_node_add_0_0 = _Retval[T=DT_FLOAT, index=0, _device="/job:localhost/replica:0/task:0/cpu:0"](node_add) is dead: 0
# execution order: x -> W -> node_mul -> B -> node_add

W = tf.constant([1.0, 2.0], name='W')
x = tf.constant([[3.0], [5.0]], name='x')
node_mul = tf.multiply(W, x, name='node_mul')

with tf.control_dependencies([node_mul]):
    B = tf.constant(0.1, name='B')

node_add = tf.add(node_mul, B, name='node_add')

with tf.Session() as sess:
    sess.run(node_add)
