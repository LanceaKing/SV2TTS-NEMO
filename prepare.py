import argparse
import json
import os
import posixpath

import kaggle
import wget


def manifest_join_base(base, manifest_file, output_file):
    with open(manifest_file, 'r') as manifest, open(output_file, 'w') as fout:
        for line in manifest:
            entry = json.loads(line)
            entry['audio_filepath'] = posixpath.join(base, entry['audio_filepath'])
            fout.write(json.dumps(entry) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True, type=str)
    args = parser.parse_args()

    prepare_dir = 'prepare'
    os.makedirs(prepare_dir, exist_ok=True)

    manifest_filepath_list = [
        os.path.join(args.input_dir, 'test-text.json'),
        os.path.join(args.input_dir, 'dev-text.json'),
        os.path.join(args.input_dir, 'train-text.json')
    ]
    for manifest_filepath in manifest_filepath_list:
        prepare_manifest_filepath = os.path.join(prepare_dir, os.path.basename(manifest_filepath))
        manifest_join_base(args.input_dir, manifest_filepath, prepare_manifest_filepath)

    r = kaggle.api.kernel_output('victorling', 'sv2tts-speaker-embeddings')
    for entry in r['files']:
        url = entry['url']
        wget.download(url, os.path.join(prepare_dir, os.path.basename(url)))


if __name__ == '__main__':
    main()
