import torch
import argparse
import json
import sys
sys.path.append("coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def evaluate():
    pass


def language_eval(dataset, preds, preds_n):
    results = {}

    if len(preds_n) > 0:
        if 'coco' in dataset:
            dataset_file = 'data/dataset_coco.json'
        elif 'flickr30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        elif 'flickr8k' in dataset:
            dataset_file = 'data/dataset_flickr8k.json'
        else:
            pass

        with open(dataset_file) as f:
            annotations = json.load(f)['images']
        
        for annot in annotations:
            if not annot['split'] in ['val']


def main(args):
    dataset = 'coco'

    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    elif 'flickr8k' in dataset:
        annFile = 'data/'
    elif 'conceptualcaption' in dataset:
        annFile = 'data/'
    else:
        pass

    coco = COCO(annFile)

    language_eval(dataset, [], [1, 2])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    main(args)