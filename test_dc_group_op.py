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

grad = [tf.constant([5,5,5], dtype=tf.float32),
        tf.constant([7,7,7], dtype=tf.float32),]
weight = [tf.Variable([2, 3, 4], dtype=tf.float32, name="grad", trainable=True),
        tf.Variable([3, 4, 5], dtype=tf.float32, name="grad", trainable=True)]
prev_weight = [tf.Variable([1, 2, 3], dtype=tf.float32, name="prev_weight", trainable=False),
              tf.Variable([2, 3, 4], dtype=tf.float32, name="prev_weight", trainable=False)]

dc_module = tf.load_op_library('./dc_op.so')

# with tf.Graph().as_default():
with tf.compat.v1.Session() as sess:

    tf.compat.v1.global_variables_initializer().run()

    prev_weight_val = sess.run(prev_weight)
    weight_val = sess.run(weight)
    grad_val = sess.run(grad)
    print('\n\n')
    print(f"Before DC OP\n\tweight \t\t= {weight_val}\n\tprev_weight \t= {prev_weight_val}")
    print(f"\tgrad \t\t=", grad_val)

    res = dc_module.delay_compensation_group(grad, [x.handle for x in prev_weight], weight, 0.5)

    updated_grad = sess.run(res)
    prev_weight_val = sess.run(prev_weight)
    weight_val = sess.run(weight)
    print(f"After DC OP\n\tweight \t\t= {weight_val}\n\tprev_weight \t= {prev_weight_val}")
    print("\tupdated_grad \t=", updated_grad)
    # print("after applying op, a =", a.eval(), "b =", b.eval())

# print(f"{a=} {b=} {c=} {d=}")

# Prints
# array([[1, 0], [0, 0]], dtype=int32)
