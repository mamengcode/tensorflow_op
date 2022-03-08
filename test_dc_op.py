import tensorflow.compat.v1 as tf1
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variables

tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_resource_variables()


# zero_out_module = tf.load_op_library('./zero_out.so')
# a = [1,2]
# b = [3,4]
# print(zero_out_module.zero_out(a))

# a = tf.Variable([1,2], dtype=np.int32, name="a", trainable=False)
# print(a.dtype._is_ref_dtype)
# x = tf.compat.v1.get_variable('x', shape=(2,), dtype=tf.float32)
# print(x.dtype)
# b = tf.Variable(np.array([3,4,5], dtype=np.int32), name="b", trainable=False)
# c = [10, 11, 12, 14]
# d = [20, 22, 24, 38,2]

grad = tf.constant([5,5,5], dtype=tf.float32)
weight = tf1.Variable([2, 3, 4], dtype=tf.float32, name="grad", trainable=True)
prev_weight = tf1.Variable([1, 2, 3], dtype=tf.float32, name="prev_weight", trainable=False)

dc_module = tf.load_op_library('./dc_op.so')

# with tf.Graph().as_default():
with tf.compat.v1.Session() as sess:

    tf.compat.v1.global_variables_initializer().run()

    print(f"before DC OP, weight = {weight.eval()}, prev_weight = {prev_weight.eval()}")
    res = dc_module.delay_compensation(grad, prev_weight.handle, weight, 0.5)
    print(sess.run([res]))
    print(f"after DC OP, weight = {weight.eval()}, prev_weight = {prev_weight.eval()}")
    # print("after applying op, a =", a.eval(), "b =", b.eval())

# print(f"{a=} {b=} {c=} {d=}")

# Prints
# array([[1, 0], [0, 0]], dtype=int32)
