# Image Captioning (WIP)


## Supported Models

* [ClipCap](https://arxiv.org/abs/2111.09734)
* [LATGeO](https://arxiv.org/abs/2109.07799) (May be Later)

## Supported Datasets

Dataset | Train | Val | Test | Captions / Image | Vocab Size | Avg. Tokens / Caption
--- | --- | --- | --- | --- | --- | ---
[COCO](https://cocodataset.org/#home) | 83k | 5k | 5k | 5 | - | -
[ConceptualCaptions](https://ai.google.com/research/ConceptualCaptions) | 3M | 15k | - | 1 | 51k | 10.3
[Flickr8k](https://forms.illinois.edu/sec/1713398) | - | - | - | - | - | -
[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) | - | - | - | - | - | -
[nocaps](https://nocaps.org/) | - | - | - | - | - | -

## Benchmarks

COCO

Model | BLEU-4↑ | METEOR↑ | ROUGE-L↑ | CIDEr↑ | SPICE↑ | Params <br><sup>(M) | Pretrained
--- | --- | --- | --- | --- | --- | --- | --- 
ClipCap | 32.2 | 27.1 | - | 108.4 | 20.1 | 156 | [download](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing)
LATGeO | 36.4 | 27.8 | 56.7 | 115.8 | - | -

Conceptual Captions

Model | ROUGE-L↑ | CIDEr↑ | SPICE↑ | Params <br><sup>(M) | Pretrained
--- | --- | --- | --- | --- | ---
ClipCap | 26.7 | 87.3 | 18.5 | 156 | [download](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing)

nocaps

Model | in-domain <br><sup>(CIDEr↑ / SPICE↑) | near-domain <br><sup>(CIDEr↑ / SPICE↑) | out-of-domain <br><sup>(CIDEr↑ / SPICE↑) | overall <br><sup>(CIDEr↑ / SPICE↑) | Params <br><sup>(M)
--- | --- | --- | --- | --- | ---
ClipCap | 79.7/12.2 | 67.7/11.3 | 49.4/9.7 | 65.7/11.1 | 156

> Notes: All these results are without CIDEr optimization.

## Requirements

* torch >= 1.8.1
* torchvision >= 0.8.1
* Python >= 3.8

Clone the repo recursively:

```bash
$ git clone --recursive https://github.com/sithu31296/image-captioning.git
```

Follow the installation steps in [coco-caption](https://github.com/sithu31296/coco-caption) if you want to evaluate, otherwise not needed.

Other requirements can be installed with:

```bash
pip install -r requirements.txt
```


## Inference

```bash
$ python tools/infer.py \
  --model-path MODEL_WEIGHTS \
  --img-path TEST_IMAGE_PATH
  --beam-search False
```

Sample inference result:

![test](assets/test.jpg)
<br>
A couple of people standing next to an elephant.


## Dataset Preparation

### COCO

* Download **COCO2014** images from [here](https://cocodataset.org/#download).
* Download Karpathy splits for COCO, Flickr8k and Flickr30k from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
* Run the following command to extract image features and tokens:

```bash
$ python tools/prepare_coco.py \
  --annot-path data/dataset_coco.json \
  --dataset-path datasets/COCO2014 \
  --save-path data/COCO
```

### Conceptual Captions

* Download Training split, Validation split and Image Labels from [here](https://ai.google.com/research/ConceptualCaptions/download).
* Run the following command to download the actual images:

```bash
$ python scripts/download_conceptual.py --root datasets/ConceptualCaptions
```

* Run the following command to extract image features and tokens:

```bash
$ python tools/prepare_conceptual.py \
  --dataset-path datasets/ConceptualCaptions \
  --save-path data/ConceptualCaptions
```

## Configuration File

Create a yaml configuration file. Default configuration file can be found in [configs/defaults.yaml](configs/defaults.yaml)

This file is needed for training and evaluation.

## Training

```bash
$ python tools/train.py --cfg CONFIG_FILE.yaml
```

## Evaluation

```bash
$ python tools/val.py --cfg CONFIG_FILE.yaml
```

## References

Most of the codes are from:

* [rmokady/CLIP_prefix_caption](https://github.com/rmokady/CLIP_prefix_caption)
* [ruotianluo/ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)
* [microsoft/Oscar](https://github.com/microsoft/Oscar)

## Citations

```
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}

```