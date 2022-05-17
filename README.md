# scivision_huggingface_segmentation

Model repository for the [scivision](https://scivision.readthedocs.io/) project that enables loading of image segmentation models from [Hugging Face](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads).

<!-- Classifies images as one of the 1000 ImageNet classes. -->

Via the scivision API, the [top 10 downloaded Image Classification models from Hugging Face](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads) (of models with a model card, last updated 17th May 2022) can be installed, loaded and run. The list of models is as follows:

1. [facebook_detr_resnet_50_panoptic](https://huggingface.co/facebook/detr-resnet-50-panoptic)
2. [facebook_detr_resnet_101_panoptic](https://huggingface.co/facebook/detr-resnet-101-panoptic)
3. [Intel_dpt_large_ade](https://huggingface.co/Intel/dpt-large-ade)
4. [microsoft_beit_base_finetuned_ade_640_640](https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640)
5. [microsoft_beit_large_finetuned_ade_640_640](https://huggingface.co/microsoft/beit-large-finetuned-ade-640-640)
6. [nvidia_segformer_b0_finetuned_ade_512_512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
7. [nvidia_segformer_b5_finetuned_ade_640_640](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640)
8. [nvidia_segformer_b4_finetuned_ade_512_512](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512)
9. [nvidia_segformer_b5_finetuned_cityscapes_1024_1024](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024)
10. [nvidia_segformer_b1_finetuned_ade_512_512](https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512)

Models in this list can be loaded and used on data with a few lines of code, e.g.

```python
from scivision import load_pretrained_model
this_repo = 'https://github.com/alan-turing-institute/scivision_huggingface_segmentation'
model = load_pretrained_model(this_repo, model='facebook_detr_resnet_50_panoptic')
```

You can then use the loaded model's predict function on image data loaded via *scivision* (see the [user guide](https://scivision.readthedocs.io/en/latest/user_guide.html) for details on how data is loaded via the scivision catalog):

```python
model.predict(<image data>)
```
