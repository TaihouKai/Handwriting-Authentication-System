import json
from glob import glob
from os import path, makedirs
from tqdm import tqdm
import base64
from argparse import ArgumentParser

if __name__ == "__main__":
    p = ArgumentParser("convert json to images")
    p.add_argument('--input', '-i')
    p.add_argument('--output', '-o')
    args = p.parse_args()
    makedirs(args.output, exist_ok=True)
    for f in glob(path.join(args.input, '*.txt')):
        fname = path.splitext(path.split(f)[-1])[0]
        j = json.load(open(f))
        ext, data = j['canvas'].split(',')
        ext = ext.split(';')[0].split('/')[-1]
        with open(path.join(args.output, f"{fname}.{ext}"), 'wb') as f:
            f.write(base64.b64decode(data))
        