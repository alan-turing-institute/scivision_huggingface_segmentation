import numpy as np
from PIL import Image
import torch
from transformers import (DetrFeatureExtractor,
                          DetrForSegmentation,
                          DPTFeatureExtractor,
                          DPTForSemanticSegmentation,
                          BeitFeatureExtractor,
                          BeitForSemanticSegmentation,
                          SegformerFeatureExtractor,
                          SegformerForSemanticSegmentation
                          )


def tidy_predict(self, image: np.ndarray) -> str:
    """Gives the top prediction for the provided image"""
    pillow_image = Image.fromarray(image.to_numpy(), 'RGB')
    inputs = self.feature_extractor(images=pillow_image, return_tensors="pt")
    outputs = self.pretrained_model(**inputs)
    logits = outputs.logits
    # model predicts COCO classes, bounding boxes, and masks
    bboxes = outputs.pred_boxes
    masks = outputs.pred_masks
    return bboxes, masks
    
    
def build_detr_model(model_name: str):
    model = DetrForSegmentation.from_pretrained(model_name)
    features = DetrFeatureExtractor.from_pretrained(model_name)
    return model, features
    
    
def build_dpt_model(model_name: str):
    model = DPTForSemanticSegmentation.from_pretrained(model_name)
    features = DPTFeatureExtractor.from_pretrained(model_name)
    return model, features
    
    
def build_beit_model(model_name: str):
    model = BeitForSemanticSegmentation.from_pretrained(model_name)
    features = BeitFeatureExtractor.from_pretrained(model_name)
    return model, features
    
    
def build_segformer_model(model_name: str):
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    features = SegformerFeatureExtractor.from_pretrained(model_name)
    return model, features

    
class facebook_detr_resnet_50_panoptic:
    def __init__(self):
        self.model_name = 'facebook/detr-resnet-50-panoptic'
        self.pretrained_model, self.feature_extractor = build_detr_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class facebook_detr_resnet_101_panoptic:
    def __init__(self):
        self.model_name = 'facebook/detr-resnet-101-panoptic'
        self.pretrained_model, self.feature_extractor = build_detr_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class Intel_dpt_large_ade:
    def __init__(self):
        self.model_name = 'Intel/dpt-large-ade'
        self.pretrained_model, self.feature_extractor = build_dpt_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class microsoft_beit_base_finetuned_ade_640_640:
    def __init__(self):
        self.model_name = 'microsoft/beit-base-finetuned-ade-640-640'
        self.pretrained_model, self.feature_extractor = build_beit_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


class microsoft_beit_large_finetuned_ade_640_640:
    def __init__(self):
        self.model_name = 'microsoft/beit-large-finetuned-ade-640-640'
        self.pretrained_model, self.feature_extractor = build_beit_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


class nvidia_segformer_b0_finetuned_ade_512_512:
    def __init__(self):
        self.model_name = 'nvidia/segformer-b0-finetuned-ade-512-512'
        self.pretrained_model, self.feature_extractor = build_segformer_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class nvidia_segformer_b5_finetuned_ade_640_640:
    def __init__(self):
        self.model_name = 'nvidia/segformer-b5-finetuned-ade-640-640'
        self.pretrained_model, self.feature_extractor = build_segformer_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


class nvidia_segformer_b4_finetuned_ade_512_512:
    def __init__(self):
        self.model_name = 'nvidia/segformer-b4-finetuned-ade-512-512'
        self.pretrained_model, self.feature_extractor = build_segformer_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class nvidia_segformer_b5_finetuned_cityscapes_1024_1024:
    def __init__(self):
        self.model_name = 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024'
        self.pretrained_model, self.feature_extractor = build_segformer_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


class nvidia_segformer_b1_finetuned_ade_512_512:
    def __init__(self):
        self.model_name = 'nvidia/segformer-b1-finetuned-ade-512-512'
        self.pretrained_model, self.feature_extractor = build_segformer_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


if __name__ == "__main__":
    pass