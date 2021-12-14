import torch
import clip
import argparse
import torch.nn.functional as F
from PIL import Image
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

import sys
sys.path.insert(0, '.')
from imgcap.models import ClipCap


@torch.inference_mode()
def generate_beam(model, tokenizer, embed, beam_size=5, entry_length=67, temp=1.0, stop_token='.'):
    stop_token_index = tokenizer.encode(stop_token)[0]

    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    for i in range(entry_length):
        outputs = model.gpt(inputs_embeds=embed)
        logits = outputs.logits
        logits = logits[:, -1, :] / temp 
        logits = logits.softmax(-1).log()

        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            embed = embed.expand(beam_size, *embed.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze()

            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat([tokens, next_tokens], dim=1)
        else:
            logits[is_stopped] = -float("Inf")
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
            next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat([tokens, next_tokens], dim=1)
            embed = embed[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]

        next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(embed.shape[0], 1, -1)
        embed = torch.cat([embed, next_token_embed], dim=1)
        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()

        if is_stopped.all():
            break
    
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts[0]
    

@torch.inference_mode()
def generate_captions(model, tokenizer, embed=None, entry_length=67, threshold=0.8, stop_token='.'):
    tokens = None
    filter_value = -float("Inf")
    stop_token_index = tokenizer.encode(stop_token)[0]

    for i in range(entry_length):
        out = model.gpt(inputs_embeds=embed)
        logits = out.logits
        logits = logits[:, -1, :] / 1.0
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_remove = cum_probs > threshold
        sorted_indices_remove[..., 1:] = sorted_indices_remove[..., :-1].clone()
        sorted_indices_remove[..., 0] = 0

        indices_remove = sorted_indices[sorted_indices_remove]
        logits[:, indices_remove] = filter_value

        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.gpt.transformer.wte(next_token)

        if tokens is None:
            tokens = next_token
        else:
            tokens = torch.cat([tokens, next_token], dim=1)
        
        embed = torch.cat([embed, next_token_embed], dim=1)

        if stop_token_index == next_token.item():
            break
    
    outputs = list(tokens.squeeze().cpu().numpy())
    output_text = tokenizer.decode(outputs)
    return output_text


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