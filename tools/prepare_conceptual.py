import torch
import argparse
import clip
import pickle
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer


class ConceptualCaptions:
    def __init__(self, dataset_path: str, save_path: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


    def prepare(self, split: str):
        data = []
        with open(self.dataset_path / f"{split}.txt") as f:
            annotations = f.read().splitlines()

        for annot in tqdm(annotations):
            fname, caption = annot.split(' ', maxsplit=1)
            image_path = self.dataset_path / split / f"{fname}.jpg"

            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                image_features = self.model.encode_image(image).cpu()

            token = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)

            data.append({
                "image_id": int(fname),
                "embedding": image_features,
                "caption": caption,
                "token": token
            })

        with open(self.save_path / f"conceptual_caption_{split}.pkl", 'wb') as f:
            pickle.dump(data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='C:\\Users\\sithu\\Documents\\Datasets\\ConceptualCaptions')
    parser.add_argument('--save-path', type=str, default='data/ConceptualCaptions')
    args = parser.parse_args()
    
    dataset = ConceptualCaptions(args.dataset_path, args.save_path)

    for split in ['val', 'train']:
        dataset.prepare(split)