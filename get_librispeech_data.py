# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# USAGE: python get_librispeech_data.py --data_root=<where to put data>
#        --data_set=<datasets_to_download> --num_workers=<number of parallel workers>
# where <datasets_to_download> can be: dev_clean, dev_other, test_clean,
# test_other, train_clean_100, train_clean_360, train_other_500 or ALL
# You can also put more than one data_set comma-separated:
# --data_set=dev_clean,train_clean_100
import argparse
import functools
import json
import logging
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sox
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LibriSpeech Data download')
parser.add_argument('--data_root', required=True, default=None, type=Path)
parser.add_argument('--data_sets', default='dev_clean', type=str)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()

URLS = {
    'TRAIN_CLEAN_100': 'http://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz',
    'TRAIN_CLEAN_360': 'http://openslr.magicdatatech.com/resources/12/train-clean-360.tar.gz',
    'TRAIN_OTHER_500': 'http://openslr.magicdatatech.com/resources/12/train-other-500.tar.gz',
    'DEV_CLEAN': 'https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz',
    'DEV_OTHER': 'http://openslr.magicdatatech.com/resources/12/dev-other.tar.gz',
    'TEST_CLEAN': 'http://openslr.magicdatatech.com/resources/12/test-clean.tar.gz',
    'TEST_OTHER': 'http://openslr.magicdatatech.com/resources/12/test-other.tar.gz',
    'DEV_CLEAN_2': 'https://openslr.magicdatatech.com/resources/31/dev-clean-2.tar.gz',
    'TRAIN_CLEAN_5': 'https://openslr.magicdatatech.com/resources/31/train-clean-5.tar.gz',
}


def __maybe_download_file(destination: Path, source: str):
    source = URLS[source]
    if not destination.exists():
        logging.info('{0} does not exist. Downloading ...'.format(destination))
        tmp = destination.with_suffix(destination.suffix + '.tmp')
        urllib.request.urlretrieve(source, filename=tmp)
        tmp.rename(destination)
        logging.info('Downloaded {0}.'.format(destination))
    else:
        logging.info('Destination {0} exists. Skipping.'.format(destination))
    return destination


def __extract_file(filepath: Path, data_dir: Path):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __process_transcript_speaker(file_path: Path, dst_folder: Path):
    entries = []
    root = file_path.parent
    transformer = sox.Transformer()
    with file_path.open(encoding='utf-8') as fin:
        for line in fin:
            id = line[:line.index(' ')]
            speaker = id[: id.index('-')]

            # Convert FLAC file to WAV
            flac_file = root / (id + '.flac')
            wav_file = dst_folder / (id + '.wav')
            if not wav_file.exists():
                transformer.build(str(flac_file), str(wav_file))

            entry = {}
            entry['audio_filepath'] = str(wav_file.absolute())
            entry['duration'] = sox.file_info.duration(wav_file)
            entry['label'] = speaker
            entries.append(entry)
    return entries


def __process_transcript_text(file_path: Path, dst_folder: Path):
    entries = []
    root = file_path.parent
    transformer = sox.Transformer()
    transformer.set_output_format(rate=22050)
    with file_path.open(encoding='utf-8') as fin:
        for line in fin:
            id, text = line[:line.index(' ')], line[line.index(' ') + 1:]
            transcript_text = text.lower().strip()

            # Convert FLAC file to WAV
            flac_file = root / (id + '.flac')
            wav_file = dst_folder / (id + '.wav')
            if not wav_file.exists():
                transformer.build(str(flac_file), str(wav_file))

            entry = {}
            entry['audio_filepath'] = str(wav_file.absolute())
            entry['duration'] = sox.file_info.duration(wav_file)
            entry['text'] = transcript_text
            entries.append(entry)
    return entries


def __process_data(data_type: str, data_folder: Path, dst_folder: Path, manifest_file: Path, num_workers: int):
    process_func = {
        'speaker': __process_transcript_speaker,
        'text': __process_transcript_text
    }

    if not dst_folder.exists():
        dst_folder.mkdir(parents=True)

    files = list(data_folder.glob('**/*.trans.txt'))

    entries = []

    with ThreadPoolExecutor(num_workers) as executor:
        processing_func = functools.partial(process_func[data_type], dst_folder=dst_folder)
        results = executor.map(processing_func, files)
        for result in tqdm(results, total=len(files)):
            entries.extend(result)

    with manifest_file.open('w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    data_sets = args.data_sets
    num_workers = args.num_workers

    if data_sets == 'ALL':
        data_sets = 'dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other'
    if data_sets == 'mini':
        data_sets = 'dev_clean_2,train_clean_5'
    for data_set in data_sets.split(','):
        logging.info('\n\nWorking on: {0}'.format(data_set))
        filepath = data_root / (data_set + '.tar.gz')
        logging.info('Getting {0}'.format(data_set))
        __maybe_download_file(filepath, data_set.upper())
        logging.info('Extracting {0}'.format(data_set))
        __extract_file(filepath, data_root)
        logging.info('Processing {0}'.format(data_set))
        __process_data(
            'speaker',
            data_root / 'LibriSpeech' / data_set.replace('_', '-'),
            data_root / 'LibriSpeech' / (data_set.replace('_', '-') + '-speaker'),
            data_root / (data_set + '_speaker.json'),
            num_workers=num_workers,
        )
        __process_data(
            'text',
            data_root / 'LibriSpeech' / data_set.replace('_', '-'),
            data_root / 'LibriSpeech' / (data_set.replace('_', '-') + '-text'),
            data_root / (data_set + '_text.json'),
            num_workers=num_workers,
        )
    logging.info('Done!')


if __name__ == '__main__':
    main()
