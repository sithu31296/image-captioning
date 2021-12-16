import torch
import time
import yaml
import argparse
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, 'imgcap')
from imgcap.models import ClipCap
from imgcap.dataset import CaptionDataset
from imgcap.utils.utils import fix_seeds


def train(model, dataloader, optimizer, scheduler, loss_fn, device, feat_len, batch_size, epoch, epochs, lr):
    model.train()
    iters_per_epoch = len(dataloader.dataset) // batch_size
    loss_per_epoch = 0.0

    pbar = tqdm(enumerate(dataloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss_per_epoch:.8f}")

    for iter, (tokens, mask, img_feat, _, _) in pbar:
        optimizer.zero_grad()

        tokens = tokens.to(device)
        mask = mask.to(device)
        img_feat = img_feat.to(device)

        outputs = model(tokens, img_feat, mask)
        logits = outputs.logits[:, feat_len-1:-1]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tokens.flatten())

        loss.backward()
        optimizer.step()
        scheduler.step()

        lr = scheduler.get_lr()
        loss_per_epoch += loss.item()

        pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss_per_epoch / (iter+1):.8f}")


def evaluate(model, dataloader, loss_fn, device, feat_len):
    model.eval()
    avg_loss = 0.0

    for tokens, mask, img_feat, _, _ in dataloader:
        tokens = tokens.to(device)
        mask = mask.to(device)
        img_feat = img_feat.to(device)

        outputs = model(tokens, img_feat, mask)
        logits = outputs.logits[:, feat_len-1:-1]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tokens.flatten())

        avg_loss += loss.item()

    return avg_loss / len(dataloader.dataset)


def main(cfg, save_dir):
    start = time.time()
    best_loss = 0.0
    epochs = cfg['EPOCHS']
    num_workers = mp.cpu_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = CaptionDataset(cfg['TRAIN_DATA_PATH'], cfg['SEQ_LENGTH'])
    valset = CaptionDataset(cfg['VAL_DATA_PATH'], cfg['SEQ_LENGTH'])

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=cfg['BATCH_SIZE'],
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    valloader = DataLoader(
        dataset=valset,
        batch_size=cfg['BATCH_SIZE'],
        sampler=None,
        num_workers=num_workers,
        pin_memory=True
    )

    model = ClipCap(cfg['SEQ_LENGTH'])
    model = model.to(device)

    optimizer = AdamW(model.parameters(), cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, scheduler, loss_fn, device, cfg['SEQ_LENGTH'], cfg['BATCH_SIZE'], epoch, epochs, cfg['LR'])

        if (epoch+1) % cfg['EVAL_INTERVAL'] == 0 or epoch == epochs-1:
            val_loss = evaluate(model, valloader, loss_fn, device, cfg['SEQ_LENGTH'])

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_dir / f"{cfg['MODEL']}_{cfg['DATASET']}.pth")
            print(f"Current Val Loss: {val_loss} Best Val Loss: {best_loss}")
    
    end = time.gmtime(time.time() - start)

    table = [
        ['Best Loss', f"{best_loss:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]

    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, save_dir)