import importlib
import os
from pathlib import Path
import sys

from src.constants import ENCODER_DIR
from src.utils.midi_encode import fetch_encoder, MIDIData
from src.utils.system import ensure_dir_exists

import torch.nn as nn


def run(args):
    # File/directory containing encodings
    enc_file = args.file
    enc_dir = args.dir

    if not (bool(enc_file) ^ bool(enc_dir)):
        print('Please specify a --dir xor --file')
        sys.exit(1)

    encoder = fetch_encoder(args.encoder)()

    if enc_dir:
        enc_files = [Path(enc_dir)/fname for fname in os.listdir(enc_dir)]
    else:
        # Encode a single MIDI file
        enc_files = [Path(enc_file)]
    dest = args.dest
    ensure_dir_exists(dest)
    MIDIData.decode_files(enc_files, dest, encoder, args.tempo)
