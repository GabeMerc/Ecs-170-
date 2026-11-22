"""Run from repository root: python -m script.stage_4_script.script_inspect_text --data-root PATH"""
import argparse
import json
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader


def main():
    parser = argparse.ArgumentParser(description='Inspect both Stage 4 datasets')
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    print(json.dumps(Dataset_Loader(data_root=args.data_root).inspect(), indent=2))


if __name__ == '__main__':
    main()
