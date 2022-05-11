import numpy as np
from PIL import Image
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation


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

    
class facebook_detr_resnet_50_panoptic:
    def __init__(self):
        self.model_name = 'facebook/detr-resnet-50-panoptic'
        self.pretrained_model, self.feature_extractor = build_detr_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


if __name__ == "__main__":
    pass