# TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
# TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# echo $TF_CFLAGS
# echo $TF_LFLAGS

TF_CFLAGS=("-I/Users/admin/.pyenv/versions/tfop/lib/python3.8/site-packages/tensorflow/include" "-D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_MAX_ALIGN_BYTES=64")
TF_LFLAGS=("-L/Users/admin/.pyenv/versions/tfop/lib/python3.8/site-packages/tensorflow" "-ltensorflow_framework.2")


file="dc_op"

g++ -std=c++14 -shared ${file}.cc -o ${file}.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -undefined dynamic_lookup

echo "Build complete"