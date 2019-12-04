import os
from pathlib import Path
import sys

from src.utils.midi_encode import decode_files
from src.utils.system import ensure_dir_exists


def run(args):
    # File/directory containing encodings
    enc_file = args.file
    enc_dir = args.dir

    if not (bool(enc_file) ^ bool(enc_dir)):
        print('Please specify a --dir xor --file')
        sys.exit(1)

    if enc_dir:
        enc_files = [Path(enc_dir)/fname for fname in os.listdir(enc_dir)]
    else:
        # Encode a single MIDI file
        enc_files = [Path(enc_file)]
    dest = args.dest
    ensure_dir_exists(dest)
    decode_files(enc_files, dest)
