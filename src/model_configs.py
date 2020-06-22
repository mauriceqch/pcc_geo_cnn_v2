from enum import Enum

from model_transforms import TransformType
from model_types import ModelType


class ModelConfig:
    def __init__(self, model_type: ModelType, model_params):
        self.model_type = model_type
        self.model_params = model_params

    def build(self):
        return self.model_type.value(**self.model_params)


class ModelConfigType(Enum):
    c1 = ModelConfig(ModelType.v1, {
        'num_filters': 32,
        'analysis_transform_type': TransformType.AnalysisTransformV1,
        'synthesis_transform_type': TransformType.SynthesisTransformV1
    })
    c2 = ModelConfig(ModelType.v2, {
        'num_filters': 32,
        'analysis_transform_type': TransformType.AnalysisTransformV1,
        'synthesis_transform_type': TransformType.SynthesisTransformV1,
        'hyper_analysis_transform_type': TransformType.HyperAnalysisTransform,
        'hyper_synthesis_transform_type': TransformType.HyperSynthesisTransform
    })
    c3 = ModelConfig(ModelType.v2, {
        'num_filters': 32,
        'analysis_transform_type': TransformType.AnalysisTransformV2,
        'synthesis_transform_type': TransformType.SynthesisTransformV2,
        'hyper_analysis_transform_type': TransformType.HyperAnalysisTransform,
        'hyper_synthesis_transform_type': TransformType.HyperSynthesisTransform
    })
    c3p = ModelConfig(ModelType.v2, {
        'num_filters': 64,
        'analysis_transform_type': TransformType.AnalysisTransformProgressiveV2,
        'synthesis_transform_type': TransformType.SynthesisTransformProgressiveV2,
        'hyper_analysis_transform_type': TransformType.HyperAnalysisTransform,
        'hyper_synthesis_transform_type': TransformType.HyperSynthesisTransform
    })

    @staticmethod
    def keys():
        return ModelConfigType.__members__.keys()

    def build(self):
        return self.value.build()
