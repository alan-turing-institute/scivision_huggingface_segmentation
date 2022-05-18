import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
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
    im = numpy.array(pillow_image)
    im = im[:, :, ::-1].copy() # Convert RGB to BGR 
    inputs = self.feature_extractor(images=pillow_image, return_tensors="pt")
    outputs = self.pretrained_model(**inputs)
    logits = outputs.logits

    v = Visualizer(im[:, :, ::-1], scale=1.5, instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = cv2.cvtColor(v.get_image()[:, :, :], cv2.COLOR_BGR2RGB)

    plot_predictions = plt.figure(figsize=(15,15))
    plt.imshow(Image.fromarray(image))
    plt.title('Predictions',fontsize='xx-large')
    plt.axis('off')
    plt.show()
    
    
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