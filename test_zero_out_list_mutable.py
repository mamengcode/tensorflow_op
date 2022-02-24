import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

# zero_out_module = tf.load_op_library('./zero_out.so')
# a = [1,2]
# b = [3,4]
# print(zero_out_module.zero_out(a))

a = tf.Variable([1,2], dtype=np.int32, name="a", trainable=False)
print(a.dtype._is_ref_dtype)
# x = tf.compat.v1.get_variable('x', shape=(2,), dtype=tf.float32)
# print(x.dtype)
# b = tf.Variable(np.array([3,4,5], dtype=np.int32), name="b", trainable=False)
# c = [10, 11, 12, 14]
# d = [20, 22, 24, 38,2]

zero_out_list_mutable_module = tf.load_op_library('./zero_out_list_mutable.so')

with tf.compat.v1.Session() as sess:
	tf.compat.v1.global_variables_initializer().run()
	update = a.assign([2,3])
	sess.run(update)
	print("after assignment, a =", a.eval())
	res = zero_out_list_mutable_module.zero_out_list_mutable(a)
	print(sess.run([res]))

print(f"{a=} {b=} {c=} {d=}")

# Prints
# array([[1, 0], [0, 0]], dtype=int32)