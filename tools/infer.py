import torch
import clip
import argparse
from PIL import Image
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

import sys
sys.path.insert(0, '.')
from imgcap.models import ClipCap
from imgcap.decode import generate_beam, generate_captions


class Inference:
    def __init__(self, model_path: str) -> None:
        self.feat_len = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.model = ClipCap(self.feat_len)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, img_path: str, beam_search: bool) -> str:
        image = Image.open(img_path)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            img_feat = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            img_embed = self.model.clip_project(img_feat).reshape(1, self.feat_len, -1)

        if beam_search:
            return generate_beam(self.model, self.tokenizer, img_embed)
            
        return generate_captions(self.model, self.tokenizer, img_embed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/clipcap/coco_weights.pt')
    parser.add_argument('--img-path', type=str, default='assets/test.jpg')
    parser.add_argument('--beam-search', type=bool, default=False)
    args = parser.parse_args()

    session = Inference(args.model_path)
    caption = session.predict(args.img_path, args.beam_search)
    print(caption)