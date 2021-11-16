import argparse
import os

import kaggle
import wget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username', required=True, type=str)
    parser.add_argument('-s', '--slug', required=True, type=str)
    parser.add_argument('-o', '--output-dir', type=str, default='prepare')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    r = kaggle.api.kernel_output(args.username, args.slug)
    for entry in r['files']:
        url = entry['url']
        wget.download(url, os.path.join(args.output_dir, os.path.basename(url)))


if __name__ == '__main__':
    main()
