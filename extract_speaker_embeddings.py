# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a helper script to extract speaker embeddings based on manifest file
Usage:
python extract_speaker_embeddings.py --manifest=/path/to/manifest/file' 
--model_path='/path/to/.nemo/file'(optional)
--embedding_dir='/path/to/embedding/directory'

Args:
--manifest: path to manifest file containing audio_file paths for which embeddings need to be extracted
--model_path(optional): path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would 
    be downloaded from NGC and used to extract embeddings
--embeddings_dir(optional): path to directory where embeddings need to stored default:'./'


"""

import json
import os
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def get_embeddings(speaker_model, manifest_file, batch_size=1, embedding_dir='./', device='cuda'):
    test_config = OmegaConf.create(
        dict(
            manifest_filepath=str(manifest_file),
            sample_rate=16000,
            labels=None,
            batch_size=batch_size,
            shuffle=False,
            time_length=20,
        )
    )

    speaker_model.setup_test_data(test_config)
    speaker_model = speaker_model.to(device)
    speaker_model.eval()

    all_embs = []
    out_embeddings = {}

    for test_batch in tqdm(speaker_model.test_dataloader()):
        test_batch = [x.to(device) for x in test_batch]
        audio_signal, audio_signal_len, labels, slices = test_batch
        with autocast():
            _, embs = speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
            emb_shape = embs.shape[-1]
            embs = embs.view(-1, emb_shape)
            all_embs.extend(embs.cpu().detach().numpy())
        del test_batch

    all_embs = np.asarray(all_embs)
    all_embs = embedding_normalize(all_embs)
    with manifest_file.open() as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            p = Path(dic['audio_filepath'])
            parts = p.relative_to(p.parents[2]).parts
            parts[1] = parts[1].replace('-speaker', '-text')
            uniq_name = '@'.join(parts)
            out_embeddings[uniq_name] = all_embs[i]

    embedding_dir = Path(embedding_dir) / 'embeddings'
    if not embedding_dir.exists():
        embedding_dir.mkdir(parents=True, exist_ok=True)

    prefix = manifest_file.stem

    embeddings_file = embedding_dir / (prefix + '_embeddings.pkl')
    pkl.dump(out_embeddings, embeddings_file.open('wb'))
    logging.info("Saved embedding files to {}".format(embedding_dir))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Path to manifest file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='ecapa_tdnn',
        required=False,
        help="path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would be downloaded from NGC and used to extract embeddings",
    )
    parser.add_argument(
        "--embedding_dir",
        type=Path,
        default=Path.cwd(),
        required=False,
        help="path to directory where embeddings need to stored default:'./'",
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.model_path.endswith('.nemo'):
        logging.info(f"Using local speaker model from {args.model_path}")
        speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(restore_path=args.model_path)
    elif args.model_path.endswith('.ckpt'):
        speaker_model = ExtractSpeakerEmbeddingsModel.load_from_checkpoint(checkpoint_path=args.model_path)
    else:
        speaker_model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name="ecapa_tdnn")
        logging.info(f"using pretrained speaker verification model from NGC")

    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
        logging.warning("Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs")

    get_embeddings(speaker_model, args.manifest, batch_size=32, embedding_dir=args.embedding_dir, device=device)


if __name__ == '__main__':
    main()
