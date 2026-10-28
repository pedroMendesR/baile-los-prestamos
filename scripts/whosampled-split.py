# Usage: python whosampled_split.py whosampled.json [-o output_dir]
# Default output: ./subsets_whosampled (samples, covers and remixes as JSON)
import argparse
import json
from pathlib import Path

from tqdm import tqdm

MAPPING = {
    'was sampled in': 'sampled',
    'sampled': 'sampled',
    'was remixed in': 'is a remix of',
    'is a remix of': 'is a remix of',
    'was covered in': 'is a cover of',
    'is a cover of': 'is a cover of',
}

OUTPUT_NAMES = {
    'sampled': 'whosampled-data-samples.json',
    'is a cover of': 'whosampled-data-covers.json',
    'is a remix of': 'whosampled-data-remixes.json',
}


def normalize(data):
    normalized = {}
    for key, item in tqdm(data.items(), desc='normalizing'):
        result = {}
        for relation in set(item.keys()):
            good = MAPPING[relation]
            if good in result:
                continue
            old = item[relation]
            if relation == good:
                result[good] = old
            else:
                result[good] = {
                    'id': key,
                    'action_name': good,
                    'track': old['other_track'],
                    'by': old['other_track_by'],
                    'other_track': old['track'],
                    'other_track_by': old['by'],
                }
        normalized[key] = result
    return normalized


def split(normalized):
    groups = {rel: {} for rel in OUTPUT_NAMES}
    for key, value in tqdm(normalized.items(), desc='splitting'):
        if len(value) != 1:
            raise ValueError(f'more than one relation for {key}: {value}')
        (relation,) = value.keys()
        groups[relation][key] = value
    return groups


def main():
    parser = argparse.ArgumentParser(
        description='Normalizes whosampled.json and splits it into samples, covers and remixes.'
    )
    parser.add_argument('input', type=Path, help='path to whosampled.json')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('subsets_whosampled'),
                        help='output directory (default: subsets_whosampled)')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'number of relations: {len(data)}')

    normalized = normalize(data)
    print(f'{len(normalized)} items normalized')

    groups = split(normalized)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for relation, filename in tqdm(OUTPUT_NAMES.items(), desc='saving'):
        content = groups[relation]
        tqdm.write(f'{len(content)} {filename}')
        with open(args.output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()