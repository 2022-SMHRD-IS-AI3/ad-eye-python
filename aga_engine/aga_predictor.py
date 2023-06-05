import torch

from .__configure import load_config
from .tools import get_transform
from .models import build_backbone, build_classifier, FeatClassifier
from PIL import Image

def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        return torch.no_grad()(fn)

    return decorate

class AGAPredictor:
    def __init__(self, logger, is_debug=True):
        self.cfg = load_config(mode='debug', logger=logger) if is_debug else load_config(mode='release', logger=logger)
        self.transform = get_transform(height=self.cfg.IMG_HEIGHT, width=self.cfg.IMG_WIDTH, is_eval=True)
        self.device = torch.device(self.cfg.DEVICE)
        self.loaded_model = False
        self.model = None

        pass
    
    def load_model(self):
        # model prefix
        backbone, c_output = build_backbone(self.cfg.BACKBONE)
        print('[AGA] backbone build complete')
        classifier = build_classifier(self.cfg.CLASSIFIER)(
            nattr=self.cfg.CLS_NATTR,
            c_in=c_output,
            bn=self.cfg.CLS_BN,
            pool=self.cfg.CLS_POOLING,
            scale =self.cfg.CLS_SCALE
        )
        print('[AGA] classifier build complete')
        self.model = FeatClassifier(backbone, classifier)
        print('[AGA] feated model build complete')
        print(self.cfg.WEIGHTS)
        # load weights
        load_dict = torch.load(self.cfg.WEIGHTS, map_location=lambda storage, loc: storage)
        state_dicts = load_dict['state_dicts']
        for key in list(state_dicts.keys()):
            state_dicts[key.replace('module.', '')] = state_dicts.pop(key)
        self.model.load_state_dict(state_dicts, strict=True)
        print('[AGA] model weight load complete')
        self.model.to(self.device).float()
        self.model.eval()
        self.loaded_model = True

    @smart_inference_mode()
    def infer_batch_once(self, imgs):
        img_tensors = []
        for img in imgs:
            img = Image.fromarray(img)
            img_tensors.append(torch.unsqueeze(self.transform(img), 0))
        # print(img_tensors[0].shape)
        imgs = torch.cat(img_tensors, 0)
        imgs = imgs.to(self.device).float()
        valid_logits, attns = self.model(imgs)
        valid_probs = torch.sigmoid(valid_logits[0])
        return valid_probs.cpu().numpy(), attns.cpu().numpy()
        
    
    