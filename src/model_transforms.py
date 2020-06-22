from enum import Enum
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Layer, Conv3D, Conv3DTranspose
from tensorflow_core.python.keras.utils import conv_utils


def get_channel_axis(data_format):
    return 1 if data_format == 'channels_first' else -1


class SequentialLayer(Layer):
    def __init__(self, layers, *args, **kwargs):
        super(SequentialLayer, self).__init__(*args, **kwargs)
        self._layers = layers

    def call(self, tensor, **kwargs):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class ResidualLayer(Layer):
    def __init__(self, layers, residual_mode='add', data_format=None, *args, **kwargs):
        super(ResidualLayer, self).__init__(*args, **kwargs)
        assert residual_mode in ('add', 'concat')
        self._layers = layers
        self.residual_mode = residual_mode
        self.data_format = conv_utils.normalize_data_format(data_format)

    def call(self, tensor, **kwargs):
        tensor = self._layers[0](tensor)
        tensor1 = tensor
        for layer in self._layers[1:]:
            tensor = layer(tensor)
        if self.residual_mode == 'add':
            return tensor1 + tensor
        else:
            return tf.concat((tensor, tensor1), get_channel_axis(self.data_format))


class AnalysisTransformV1(SequentialLayer):
    def __init__(self, filters, data_format=None, activation=tf.nn.relu, *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'strides': (2, 2, 2), 'padding': 'same', 'data_format': data_format, 'filters': filters}
        layers = [Conv3D(kernel_size=(9, 9, 9), use_bias=True, activation=activation, **params),
                  Conv3D(kernel_size=(5, 5, 5), use_bias=True, activation=activation, **params),
                  Conv3D(kernel_size=(5, 5, 5), use_bias=False, activation=None, **params)]
        super(AnalysisTransformV1, self).__init__(layers, *args, **kwargs)


class SynthesisTransformV1(SequentialLayer):
    def __init__(self, filters, data_format=None, activation=tf.nn.relu, *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'strides': (2, 2, 2), 'padding': 'same', 'data_format': data_format, 'use_bias': True,
                  'activation': activation}
        layers = [Conv3DTranspose(filters, (5, 5, 5), **params),
                  Conv3DTranspose(filters, (5, 5, 5), **params),
                  Conv3DTranspose(1, (9, 9, 9), **params)]
        super(SynthesisTransformV1, self).__init__(layers, *args, **kwargs)


class AnalysisBlock(ResidualLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation=tf.nn.relu, *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'padding': 'same', 'data_format': data_format, 'use_bias': True, 'activation': activation,
                  'filters': filters, 'kernel_size': kernel_size}
        layers = [Conv3D(strides=strides, **params),
                  Conv3D(**params),
                  Conv3D(**params)]
        super(AnalysisBlock, self).__init__(layers, *args, data_format=data_format, **kwargs)


class SynthesisBlock(ResidualLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation=tf.nn.relu, *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'padding': 'same', 'data_format': data_format, 'use_bias': True, 'activation': activation,
                  'filters': filters, 'kernel_size': kernel_size}
        layers = [Conv3DTranspose(strides=strides, **params),
                  Conv3DTranspose(**params),
                  Conv3DTranspose(**params)]
        super(SynthesisBlock, self).__init__(layers, *args, data_format=data_format, **kwargs)


class AnalysisTransformV2(SequentialLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, residual_mode='add',
                 *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'kernel_size': kernel_size, 'activation': activation, 'data_format': data_format,
                  'residual_mode': residual_mode}
        layers = [AnalysisBlock(filters // 2, **params),
                  AnalysisBlock(filters, **params),
                  AnalysisBlock(filters, **params),
                  Conv3D(filters, kernel_size, padding="same", use_bias=False, activation=None,
                         data_format=data_format)]
        super(AnalysisTransformV2, self).__init__(layers, *args, **kwargs)


class SynthesisTransformV2(SequentialLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, residual_mode='add',
                 *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'kernel_size': kernel_size, 'activation': activation, 'data_format': data_format,
                  'residual_mode': residual_mode}
        layers = [SynthesisBlock(filters, **params),
                  SynthesisBlock(filters, **params),
                  SynthesisBlock(filters // 2, **params),
                  Conv3DTranspose(1, kernel_size, padding="same", use_bias=True, activation=activation,
                                  data_format=data_format)]
        super(SynthesisTransformV2, self).__init__(layers, *args, **kwargs)


class AnalysisTransformProgressiveV2(SequentialLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, residual_mode='add',
                 *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'kernel_size': kernel_size, 'activation': activation, 'data_format': data_format,
                  'residual_mode': residual_mode}
        layers = [AnalysisBlock(filters // 4, **params),
                  AnalysisBlock(filters // 2, **params),
                  AnalysisBlock(filters, **params),
                  Conv3D(filters, kernel_size, padding="same", use_bias=False, activation=None,
                         data_format=data_format)]
        super(AnalysisTransformProgressiveV2, self).__init__(layers, *args, **kwargs)


class SynthesisTransformProgressiveV2(SequentialLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, residual_mode='add',
                 *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'kernel_size': kernel_size, 'activation': activation, 'data_format': data_format,
                  'residual_mode': residual_mode}
        layers = [SynthesisBlock(filters, **params),
                  SynthesisBlock(filters // 2, **params),
                  SynthesisBlock(filters // 4, **params),
                  Conv3DTranspose(1, kernel_size, padding="same", use_bias=True, activation=activation,
                                  data_format=data_format)]
        super(SynthesisTransformProgressiveV2, self).__init__(layers, *args, **kwargs)


class HyperAnalysisTransform(SequentialLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'padding': 'same', 'data_format': data_format, 'filters': filters, 'kernel_size': kernel_size}
        layers = [Conv3D(use_bias=True, activation=activation, **params),
                  Conv3D(use_bias=True, activation=activation, strides=(2, 2, 2), **params),
                  Conv3D(use_bias=False, activation=None, **params)]
        super(HyperAnalysisTransform, self).__init__(layers, *args, **kwargs)


class HyperSynthesisTransform(SequentialLayer):
    def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, *args, **kwargs):
        data_format = conv_utils.normalize_data_format(data_format)
        params = {'padding': 'same', 'data_format': data_format, 'activation': activation, 'use_bias': True,
                  'filters': filters, 'kernel_size': kernel_size}
        layers = [Conv3DTranspose(**params),
                  Conv3DTranspose(strides=(2, 2, 2), **params),
                  Conv3DTranspose(**params)]
        super(HyperSynthesisTransform, self).__init__(layers, *args, **kwargs)


class TransformType(Enum):
    AnalysisTransformV1 = AnalysisTransformV1
    AnalysisTransformV2 = AnalysisTransformV2
    AnalysisTransformProgressiveV2 = AnalysisTransformProgressiveV2
    SynthesisTransformV1 = SynthesisTransformV1
    SynthesisTransformV2 = SynthesisTransformV2
    SynthesisTransformProgressiveV2 = SynthesisTransformProgressiveV2
    HyperAnalysisTransform = HyperAnalysisTransform
    HyperSynthesisTransform = HyperSynthesisTransform
