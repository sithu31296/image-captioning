import torch
import argparse
import json
import clip
import pickle
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer


class COCOCaption:
    def __init__(self, annot_path: str, dataset_path: str, save_path: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        with open(annot_path) as f:
            dataset = json.load(f)
            self.annotations = dataset['images']
            self.dataset_name = dataset['dataset']


    def prepare(self, split):
        data = []
        annotations = list(filter(lambda x: x['split'] == split, self.annotations))

        print(f"Preparing {self.dataset_name} {split} split...")
        for annot in tqdm(annotations):
            caption = annot['sentences'][0]['raw']

        #     if self.dataset_name == "coco":
        #         image_path = self.dataset_path / annot['filepath'] / annot['filename']
        #         image_id = annot['cocoid']
        #     elif self.dataset_name == "flickr8k":
        #         image_path = self.dataset_path / "Images" / annot['filename']
        #         image_id = annot['imgid']
        #     elif self.dataset_name == "flickr30k":
        #         image_path = self.dataset_path / "flickr30k_images" / annot['filename']
        #         image_id = annot['imgid']

        #     image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        #     with torch.inference_mode():
        #         image_features = self.model.encode_image(image).cpu()

        #     token = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
            
        #     data.append({
        #         "image_id": image_id,
        #         "embedding": image_features,
        #         "caption": caption,
        #         "token": token
        #     })

        # with open(self.save_path / f"{self.dataset_name}_{split}.pkl", 'wb') as f:
        #     pickle.dump(data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot-path', type=str, default='data/dataset_flickr30k.json')
    parser.add_argument('--dataset-path', type=str, default='C:\\Users\\sithu\\Documents\\Datasets\\Flickr30k')
    parser.add_argument('--save-path', type=str, default='data/Flickr30k')
    args = parser.parse_args()
    
    dataset = COCOCaption(args.annot_path, args.dataset_path, args.save_path)
    
    for split in ['test', 'val', 'train']:
        dataset.prepare(split)