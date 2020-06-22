import numpy as np
import tensorflow.compat.v1 as tf
from numpy.testing import assert_array_equal

from model_transforms import AnalysisTransformV1, SynthesisTransformV1, AnalysisBlock, SynthesisBlock, \
    AnalysisTransformV2, SynthesisTransformV2, \
    HyperAnalysisTransform, HyperSynthesisTransform, AnalysisTransformProgressiveV2, SynthesisTransformProgressiveV2


def tf_to_np(ts):
    return np.array(ts.as_list())


class TestModelTransforms(tf.test.TestCase):
    def setUp(self):
        self.x = tf.constant(np.zeros((1, 8, 8, 8, 1)))
        self.y = tf.constant(np.zeros((1, 1, 1, 1, 1)))
        self.data_format = 'channels_last'

    def run_layer_test(self, layer, x):
        x = layer(x)
        with self.cached_session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            return sess.run(x)

    def test_analysis_transform_v1(self):
        x = self.run_layer_test(AnalysisTransformV1(1, data_format=self.data_format), self.x)
        assert_array_equal(x.shape, tf_to_np(self.x.shape) // [1, 8, 8, 8, 1])

    def test_synthesis_transform_v1(self):
        y = self.run_layer_test(SynthesisTransformV1(2, data_format=self.data_format), self.y)
        assert_array_equal(y.shape, tf_to_np(self.y.shape) * [1, 8, 8, 8, 1])

    def test_analysis_block(self):
        x = self.run_layer_test(AnalysisBlock(1, data_format=self.data_format), self.x)
        assert_array_equal(x.shape, tf_to_np(self.x.shape) // [1, 2, 2, 2, 1])
        x = self.run_layer_test(AnalysisBlock(1, data_format=self.data_format, residual_mode='concat'), self.x)
        assert_array_equal(x.shape, tf_to_np(self.x.shape) // [1, 2, 2, 2, 0.5])

    def test_synthesis_block(self):
        y = self.run_layer_test(SynthesisBlock(1, data_format=self.data_format), self.y)
        assert_array_equal(y.shape, tf_to_np(self.y.shape) * [1, 2, 2, 2, 1])
        y = self.run_layer_test(SynthesisBlock(1, data_format=self.data_format, residual_mode='concat'), self.y)
        assert_array_equal(y.shape, tf_to_np(self.y.shape) * [1, 2, 2, 2, 2])

    def test_analysis_transform_v2(self):
        x = self.run_layer_test(AnalysisTransformV2(2, data_format=self.data_format), self.x)
        assert_array_equal(x.shape, tf_to_np(self.x.shape) // [1, 8, 8, 8, 0.5])
        x = self.run_layer_test(AnalysisTransformV2(2, data_format=self.data_format, residual_mode='concat'), self.x)
        assert_array_equal(x.shape, tf_to_np(self.x.shape) // [1, 8, 8, 8, 0.5])

    def test_synthesis_transform_v2(self):
        y = self.run_layer_test(SynthesisTransformV2(2, data_format=self.data_format), self.y)
        assert_array_equal(y.shape, tf_to_np(self.y.shape) * [1, 8, 8, 8, 1])
        y = self.run_layer_test(SynthesisTransformV2(2, data_format=self.data_format, residual_mode='concat'), self.y)
        assert_array_equal(y.shape, tf_to_np(self.y.shape) * [1, 8, 8, 8, 1])

    def test_analysis_transform_progressive_v2(self):
        x = self.run_layer_test(AnalysisTransformProgressiveV2(4, data_format=self.data_format), self.x)
        assert_array_equal(x.shape, tf_to_np(self.x.shape) // [1, 8, 8, 8, 0.25])

    def test_synthesis_transform_progressive_v2(self):
        y = self.run_layer_test(SynthesisTransformProgressiveV2(4, data_format=self.data_format), self.y)
        assert_array_equal(y.shape, tf_to_np(self.y.shape) * [1, 8, 8, 8, 1])

    def test_hyper_analysis_transform(self):
        x = self.run_layer_test(HyperAnalysisTransform(1, data_format=self.data_format), self.x)
        assert_array_equal(x.shape, tf_to_np(self.x.shape) // [1, 2, 2, 2, 1])

    def test_hyper_synthesis_transform(self):
        y = self.run_layer_test(HyperSynthesisTransform(1, data_format=self.data_format), self.y)
        assert_array_equal(y.shape, tf_to_np(self.y.shape) * [1, 2, 2, 2, 1])

