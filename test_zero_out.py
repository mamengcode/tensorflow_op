import tensorflow as tf


# zero_out_module = tf.load_op_library('./zero_out.so')
# a = [1,2]
# b = [3,4]
# print(zero_out_module.zero_out(a))

zero_out_list_module = tf.load_op_library('./zero_out_list.so')
a = [1,2]
b = [3,4,5]
c = [10, 11, 12, 14]
d = [20, 22, 24]
e = [21, 23]
print(zero_out_list_module.zero_out_list([a, b], [d, e]))

# Prints
# array([[1, 0], [0, 0]], dtype=int32)