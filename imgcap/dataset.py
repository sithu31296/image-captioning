import torch
import pickle
from functools import reduce
from torch.utils.data import Dataset



class CaptionDataset(Dataset):
    def __init__(self, data_path: str, feat_len: int) -> None:
        super().__init__()
        self.feat_len = feat_len
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.max_seq_len = reduce(lambda x, y: max(x, len(y['token'])), self.data, 0)

    def __len__(self) -> int:
        return len(self.data)

    def pad_tokens(self, tokens):
        pad_size = self.max_seq_len - tokens.shape[0]

        if pad_size > 0:
            tokens = torch.cat([tokens, torch.zeros(pad_size, dtype=torch.int64) - 1])
        elif pad_size < 0:
            tokens = tokens[:self.max_seq_len]

        mask = tokens > 0
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat([torch.ones(self.feat_len), mask], dim=0)
        return tokens, mask

    def __getitem__(self, index):
        image_id = self.data[index]['image_id']
        caption = self.data[index]['caption']
        tokens = self.data[index]['token']
        img_feat = self.data[index]['embedding']
        tokens, mask = self.pad_tokens(tokens)
        return tokens, mask, img_feat, image_id, caption


if __name__ == '__main__':
    dataset = CaptionDataset('data/coco_oscar_split_train.pkl', 10)
    tokens, mask, feat = next(iter(dataset))
    print(tokens.shape)
    print(tokens)
    print(mask.shape)
    print(feat.shape)
