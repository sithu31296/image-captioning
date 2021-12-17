# Image Captioning 

Most Image Captioning models are complicated and very hard to test. Traditional Image caption model first encodes the image using [BUTD](https://arxiv.org/abs/1707.07998) model, called the bottom up features. This is a Faster-RCNN model trained on [Visual Genome](https://visualgenome.org/) dataset. And then use an attention or transformer model to generate a caption. There is also the use of [SCST](https://arxiv.org/abs/1612.00563) to improve the results.

In 2020, a new model from Microsoft is released, called [Oscar](https://github.com/microsoft/Oscar), which sets a new record in image captioning. This involves pre-training on large amount of datasets and fine-tuned on downstream tasks. This is also a very complicated and time-consuming process. There is also an improved work called [VinVL](https://arxiv.org/abs/2101.00529); which uses their object-attribute detection model to extract features instead of usual bottom-up features used in Oscar.

After coming out the zero-shot model [CLIP](https://arxiv.org/abs/2103.00020) from OpenAI, many papers released on vision-language related tasks like [CLIP-ViL](https://arxiv.org/abs/2107.06383), [X-modaler](https://arxiv.org/abs/2103.17249) and lastly [ClipCap](https://arxiv.org/abs/2111.09734). Among them, ClipCap is the most simplest network everyone can easily test.

## Benchmarks

COCO

Model | BLEU-4↑ | METEOR↑ | ROUGE-L↑ | CIDEr↑ | SPICE↑ | Params <br><sup>(M) | Pretrained
--- | --- | --- | --- | --- | --- | --- | --- 
[ClipCap](https://arxiv.org/abs/2111.09734) | 32.2 | 27.1 | - | 108.4 | 20.1 | 156 | [download](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing)
[LATGeO](https://arxiv.org/abs/2109.07799) | 36.4 | 27.8 | 56.7 | 115.8 | - | -

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

## Datasets

Dataset | Train | Val | Test | Captions / Image | Vocab Size | Avg. Tokens / Caption
--- | --- | --- | --- | --- | --- | ---
[COCO](https://cocodataset.org/#home) | 83k | 5k | 5k | 5 | - | -
[ConceptualCaptions](https://ai.google.com/research/ConceptualCaptions) | 3M | 15k | - | 1 | 51k | 10.3
[Flickr8k](https://forms.illinois.edu/sec/1713398) | 6k | 1k | 1k | 5 | - | -
[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) | 29k | 1k | 1k | 5 | - | -
[nocaps](https://nocaps.org/) | - | - | - | - | - | -

## Datasets Preparation

### COCO / Flickr8k / Flickr30k

* Download dataset images.
  * For COCO, download **COCO2014** images from [COCO](https://cocodataset.org/#download).
  * For Flickr8k, download images from [Official Website](https://forms.illinois.edu/sec/1713398) or if you can't download it, try downloading from [Kaggle](https://www.kaggle.com/adityajn105/flickr8k).
  * For Flickr30k, download images from [Official Website](http://shannon.cs.illinois.edu/DenotationGraph/) or if you can't download it, try downloading from [Kaggle](https://www.kaggle.com/hsankesara/flickr-image-dataset).
* Download Karpathy splits for COCO, Flickr8k and Flickr30k from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
* Run the following command to extract image features and tokens:

```bash
$ python tools/prepare_coco_flickr.py \
  --annot-path KARPATHY_ANNOT_JSON_PATH \
  --dataset-path DATASET_ROOT_PATH \
  --save-path SAVE_PATH
```

* To evaluate with `coco-caption`, you need to convert Karpathy split json format to COCO json format.

```bash
$ python scripts/convert_coco_format.py \
  --input-json KARPATHY_ANNOT_JSON_PATH \
  --output-json COCO_JSON_SAVE_PATH \
  --split 'test' or 'val'
```

> To evaluate on COCO-val, you can also use annotation file in `coco_caption/annotations/captions_val2014.json`.

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

* To evaluate with `coco-caption`, you need to convert to COCO json format.

```bash
$ python scripts/convert_conceptual_to_coco.py \
  --input-txt VAL_TXT_PATH \
  --output-json COCO_JSON_SAVE_PATH
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