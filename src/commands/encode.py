import os
from pathlib import Path

from src.utils.midi_encode import encode_midi_files, gen_enc_filename
from src.utils.system import ensure_dir_exists


def run(args):
    midi_file = args.file
    midi_dir = args.dir

    if not (bool(midi_dir) ^ bool(midi_file)):
        print('Please specify a --dir xor --file')
        sys.exit(1)

    def instrument_filter(
        instrument): return args.instrument_filter in str(instrument).lower()

    if midi_dir:
        midi_files = [Path(midi_dir)/fname for fname in os.listdir(midi_dir)
                      if Path(fname).suffix in ('.mid',)]
    else:
        # Encode a single MIDI file
        midi_files = [Path(fname)]
    dest = args.dest
    ensure_dir_exists(dest)
    encode_midi_files(
        list(midi_files), dest=dest, instrument_filter=instrument_filter, no_transpose=args.no_transpose)
