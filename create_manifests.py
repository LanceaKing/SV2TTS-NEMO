# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
Creates the manifest and .txt transcript files for an LJSpeech split.
The manifest will be used for training, and the .txt files are for the MFA library to find.
"""

import argparse
import json
from pathlib import Path

import sox
import wget
from nemo.collections.common.parts.preprocessing import parsers
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ljspeech_base', required=True, default=None, type=Path)
args = parser.parse_args()


def main():
    filelist_base = 'https://raw.githubusercontent.com/NVIDIA/tacotron2/master/filelists/'
    filelists = ['train', 'val', 'test']

    # NeMo parser for text normalization
    text_parser = parsers.make_parser(name='en')

    for split in filelists:
        # Download file list if necessary
        filelist_path = args.ljspeech_base / f"ljs_audio_text_{split}_filelist.txt"
        if not filelist_path.exists():
            wget.download(f"{filelist_base}/ljs_audio_text_{split}_filelist.txt", out=str(args.ljspeech_base))

        args.ljspeech_base.joinpath('wavs-speaker').mkdir(parents=True, exist_ok=True)
        transformer = sox.Transformer()
        transformer.set_output_format(rate=16000)
        manifest_target = args.ljspeech_base / f"ljspeech_{split}_speaker.json"
        with manifest_target.open('w', encoding='utf-8') as f_out, filelist_path.open('r', encoding='utf-8') as filelist:
            print(f"\nCreating {manifest_target}...")
            for line in tqdm(filelist.readlines()):
                basename = line[6:16]
                wav_path = args.ljspeech_base / 'wavs' / (basename + '.wav')
                out_path = args.ljspeech_base / 'wavs-speaker' / (basename + '.wav')
                if not out_path.exists():
                    transformer.build(str(wav_path), str(out_path))

                # Write manifest entry
                entry = {
                    'audio_filepath': str(out_path),
                    'duration': sox.file_info.duration(out_path),
                    'label': 'LJ',
                }
                f_out.write(json.dumps(entry) + '\n')

        manifest_target = args.ljspeech_base / f"ljspeech_{split}_text.json"
        with manifest_target.open('w', encoding='utf-8') as f_out, filelist_path.open('r', encoding='utf-8') as filelist:
            print(f"\nCreating {manifest_target}...")
            for line in tqdm(filelist.readlines()):
                basename = line[6:16]
                text = text_parser._normalize(line[21:].strip())
                wav_path = args.ljspeech_base / 'wavs' / (basename + '.wav')

                # Write manifest entry
                entry = {
                    'audio_filepath': str(wav_path),
                    'duration': sox.file_info.duration(wav_path),
                    'text': text,
                }
                f_out.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    main()
