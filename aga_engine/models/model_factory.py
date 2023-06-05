from .model_registry import BACKBONE
from .model_registry import CLASSIFIER
from .model_registry import LOSSES
from . import swin_t

model_dict = {
    'swin_t': 768,
    'swin_s': 768,
    'swin_b': 1024,

}
def build_backbone(key):
    model = BACKBONE[key]()
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

