import argparse
from pathlib import Path

import torch
from nemo.collections.tts.models import Tacotron2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modules_dir', required=False, default=Path.cwd(), type=Path)
    args = parser.parse_args()

    modules_dir = args.modules_dir / 'modules'
    if not modules_dir.exists():
        modules_dir.mkdir(parents=True, exist_ok=True)

    model = Tacotron2Model.from_pretrained('tts_en_tacotron2')
    text_embedding = model.text_embedding
    torch.save(text_embedding.state_dict(), modules_dir / 'text_embedding.pt')
    encoder = model.encoder
    torch.save(encoder.state_dict(), modules_dir / 'encoder.pt')
    postnet = model.postnet
    torch.save(postnet.state_dict(), modules_dir / 'postnet.pt')


if __name__ == '__main__':
    main()
