import os
from pathlib import Path
import sys

from src.utils.midi_encode import MIDIData, fetch_encoder, gen_enc_filename
from src.utils.system import ensure_dir_exists


def run(args):
    midi_file = args.file
    midi_dir = args.dir
    encoder = fetch_encoder(args.encoder)()

    if not (bool(midi_dir) ^ bool(midi_file)):
        print('Please specify a --dir xor --file')
        sys.exit(1)

    def instrument_filter(
        instrument): return args.filter in str(instrument).lower()

    if midi_dir:
        midi_files = [Path(midi_dir)/fname for fname in os.listdir(midi_dir)
                      if Path(fname).suffix in ('.mid',)]
    else:
        # Encode a single MIDI file
        midi_files = [Path(midi_file)]
    dest = args.dest
    ensure_dir_exists(dest)
    MIDIData.encode_files(
        list(midi_files), dest, encoder, instrument_filter=instrument_filter, no_transpose=args.no_transpose)
