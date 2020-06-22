import tensorflow as tf
import numpy as np
import tensorflow_compression as tfc
from tensorflow_compression.python.ops import range_coding_ops
from tensorflow_compression.python.ops import math_ops


def add_prefix_to_dict(prefix, d):
    return {f'{prefix}/{k}': v for k, v in d.items()}


def decompress(self, strings, **kwargs):
    with tf.name_scope(self._name_scope()):
        strings = tf.convert_to_tensor(strings, dtype=tf.string)

        indexes = self._prepare_indexes(**kwargs)
        ndim = self.input_spec.ndim
        broadcast_indexes = (indexes.shape.ndims != ndim)
        if broadcast_indexes:
            # We can't currently broadcast over anything else but the batch axis.
            assert indexes.shape.ndims == ndim - 1
            args = (strings,)
        else:
            args = (strings, indexes)

        def loop_body(args):
            symbols = range_coding_ops.unbounded_index_range_decode(
                args[0], indexes if broadcast_indexes else args[1],
                self._quantized_cdf, self._cdf_length, self._offset,
                precision=self.range_coder_precision, overflow_width=4,
                debug_level=0)
            return symbols

        symbols = tf.map_fn(
            loop_body, args, dtype=tf.int32, back_prop=False, name="decompress")

        outputs = self._dequantize(symbols, "dequantize")
        assert outputs.dtype == self.dtype

        if not tf.executing_eagerly():
            outputs.set_shape(self.input_spec.shape)

        dbg_dec = {**{'strings': strings}, **self.dbg_build, **{'indexes': indexes, 'quantized_cdf': self._quantized_cdf,
                                                                'symbols': symbols, 'outputs': outputs}}
        self.dbg_dec = add_prefix_to_dict('decompress', dbg_dec)
        return outputs


def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape.assert_is_compatible_with(self.input_spec.shape)

    scale_table = tf.constant(self.scale_table, dtype=self.dtype)

    # Lower bound scales. We need to do this here, and not in __init__, because
    # the dtype may not yet be known there.
    if self.scale_bound is None:
        self._scale = math_ops.lower_bound(self._scale, scale_table[0])
    elif self.scale_bound > 0:
        self._scale = math_ops.lower_bound(self._scale, self.scale_bound)

    multiplier = -self._standardized_quantile(self.tail_mass / 2)
    pmf_center = np.ceil(np.array(self.scale_table) * multiplier).astype(int)
    pmf_length = 2 * pmf_center + 1
    max_length = np.max(pmf_length)

    # This assumes that the standardized cumulative has the property
    # 1 - c(x) = c(-x), which means we can compute differences equivalently in
    # the left or right tail of the cumulative. The point is to only compute
    # differences in the left tail. This increases numerical stability: c(x) is
    # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
    # done with much higher precision than subtracting two numbers close to 1.
    samples = abs(np.arange(max_length, dtype=int) - pmf_center[:, None])
    samples = tf.constant(samples, dtype=self.dtype)
    samples_scale = tf.expand_dims(scale_table, 1)
    upper = self._standardized_cumulative((.5 - samples) / samples_scale)
    lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
    pmf = upper - lower

    # Compute out-of-range (tail) masses.
    tail_mass = 2 * lower[:, :1]

    def cdf_initializer(shape, dtype=None, partition_info=None):
        del partition_info  # unused
        assert tuple(shape) == (len(pmf_length), max_length + 2)
        assert dtype == tf.int32
        return self._pmf_to_cdf(
            pmf, tail_mass,
            tf.constant(pmf_length, dtype=tf.int32), max_length)

    quantized_cdf = self.add_weight(
        "quantized_cdf", shape=(len(pmf_length), max_length + 2),
        initializer=cdf_initializer, dtype=tf.int32, trainable=False)
    cdf_length = self.add_weight(
        "cdf_length", shape=(len(pmf_length),),
        initializer=tf.initializers.constant(pmf_length + 2),
        dtype=tf.int32, trainable=False)
    # Works around a weird TF issue with reading variables inside a loop.
    self._quantized_cdf = tf.identity(quantized_cdf)
    self._cdf_length = tf.identity(cdf_length)

    # Now, if they haven't been overridden, compute the indexes into the table
    # for each of the passed-in scales.
    if not hasattr(self, "_indexes"):
        # Prevent tensors from bouncing back and forth between host and GPU.
        with tf.device("/cpu:0"):
            fill = tf.constant(
                len(self.scale_table) - 1, dtype=tf.int32)
            initializer = tf.fill(tf.shape(self.scale), fill)

            def loop_body(indexes, scale):
                return indexes - tf.cast(self.scale <= scale, tf.int32)

            self._indexes = tf.foldr(
                loop_body, scale_table[:-1],
                initializer=initializer, back_prop=False, name="compute_indexes")

    self._offset = tf.constant(-pmf_center, dtype=tf.int32)

    # tfc.SymmetricConditional.build(self, input_shape)
    super(tfc.SymmetricConditional, self).build(input_shape)
    dbg_build = {'scale_table': scale_table, '_scale': self._scale, '_quantized_cdf': self._quantized_cdf,
                 '_cdf_length': cdf_length, '_offset': self._offset, 'fill': fill,
                 'initializer': initializer, '_indexes': self._indexes}
    self.dbg_build = add_prefix_to_dict('build', dbg_build)


def patch_gaussian_conditional(gaussian_conditional):
    gaussian_conditional.decompress = decompress
    gaussian_conditional.build = build
    return gaussian_conditional
