from keras.layers import Layer
from tensorflow.python.framework import dtypes, function
from tensorflow.python.ops import math_ops,array_ops,nn

class sparsemax(Layer):

    def __init__(self, **kwargs):
        super(sparsemax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(sparsemax, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        return self.compute_sparsemax(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_sparsemax(self,input_x):
        @function.Defun()
        def bprop(x, grad):
            sparsemax = prop_raw(x)
            support = math_ops.cast(sparsemax > 0, sparsemax.dtype)

            v_hat = math_ops.reduce_sum(math_ops.mul(grad, support), axis=1) \
                    / math_ops.reduce_sum(support, axis=1)

            return [support * (grad - v_hat[:, array_ops.newaxis])]

        @function.Defun()
        def prop_raw(x):
            obs = array_ops.shape(x)[0]
            dim = array_ops.shape(x)[1]

            z = x - math_ops.reduce_mean(x, axis=1)[:, array_ops.newaxis]

            z_sorted, _ = nn.top_k(z, k=dim)

            z_cumsum = math_ops.cumsum(z_sorted, axis=1)
            k = math_ops.range(
                1, math_ops.cast(dim, x.dtype) + 1, dtype=x.dtype
            )
            z_check = 1 + k * z_sorted > z_cumsum

            k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

            indices = array_ops.stack([math_ops.range(0, obs), k_z - 1], axis=1)
            tau_sum = array_ops.gather_nd(z_cumsum, indices)
            tau_z = (tau_sum - 1) / math_ops.cast(k_z, x.dtype)

            return math_ops.maximum(
                math_ops.cast(0, x.dtype),
                z - tau_z[:, array_ops.newaxis]
            )

        @function.Defun(grad_func=bprop)
        def prop(x):
            return prop_raw(x)
        return prop(input_x)
