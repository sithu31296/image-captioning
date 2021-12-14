import argparse
import requests
import shutil
from tqdm import tqdm
from pathlib import Path


def download_image(url: str, save_path: str):
    try:
        r = requests.get(url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True

            with open(save_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            return 1
        else:
            return 0
    except:
        return 0


def main(root):
    root = Path(root)

    for split in ['val', 'train']:
        save_path = root / split
        save_path.mkdir(exist_ok=True)

        tsv_path = root / ("Train_GCC-Training.tsv" if split == 'train' else "Validation_GCC-1.1.0-Validation.tsv")
    
        with open(tsv_path, encoding='utf-8') as f:
            lines = f.read().splitlines()

        print(f"Total Images to be downloaded >> {len(lines)}")

        annotations = []

        for i, line in tqdm(enumerate(lines), total=len(lines)):
            caption, url = line.split('\t')
            fname = f"{i:08d}"

            success = download_image(url, save_path / f"{fname}.jpg")

            if success:
                annotations.append(f"{fname} {caption}\n")
            else:
                print(f"\nError downloading from {url}")

        with open(root / f"{split}.txt", 'w') as f:
            f.writelines(annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='C:\\Users\\sithu\\Documents\\Datasets\\ConceptualCaptions')
    args = parser.parse_args()
    main(args.root)