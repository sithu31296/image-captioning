import torch
from torch import nn, Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class MLP(nn.Module):
    def __init__(self, feat_dim, embed_dim, feat_len) -> None:
        super().__init__()
        ch = (embed_dim * feat_len) // 2
        self.model = nn.Sequential(
            nn.Linear(feat_dim, ch),
            nn.Tanh(),
            nn.Linear(ch, embed_dim * feat_len)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ClipCap(nn.Module):
    def __init__(self, feat_len: int, feat_dim: int = 512) -> None:
        super().__init__()
        self.feat_len = feat_len
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embed_dim = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP(feat_dim, self.gpt_embed_dim, feat_len)
        
    def forward(self, tokens: Tensor, img_feat: Tensor, mask=None) -> Tensor:
        img_proj = self.clip_project(img_feat).view(-1, self.feat_len, self.gpt_embed_dim)
        text_embed = self.gpt.transformer.wte(tokens)
        embeddings = torch.cat([img_proj, text_embed], dim=1)

        out = self.gpt(
            inputs_embeds=embeddings,
            attention_mask=mask
        )
        return out


if __name__ == '__main__':
    model = ClipCap(10)
    model.load_state_dict(torch.load('checkpoints/clipcap/coco_weights.pt', map_location='cpu'))
    img_feat = torch.randn(1, 512)
    tokens = torch.randint(low=0, high=10000, size=(1, 10))
    mask = torch.ones(20)
    out = model(tokens, img_feat, mask)
