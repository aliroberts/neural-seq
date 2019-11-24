import math
import os
from random import random

import pandas as pd

from src.utils.midi_data import midi_data_required
from src.utils.midi_encode import encode_midi_files, gen_enc_filename
from src.utils.system import copyfile, ensure_dir_exists

from src.constants import MIDI_ARTISTS


@midi_data_required
def gen_dataset_from_artists(artists, dest, instrument_filter, no_transpose=False,
                             valid=0.2, test=0, skip_existing=True):
    midi_files = []
    for artist in os.listdir(MIDI_ARTISTS):
        song_dir = MIDI_ARTISTS/artist
        midi_files += [song_dir/song for song in os.listdir(song_dir)]

    # Encode all the midi files to an 'all' directory
    save_all_to = Path(dest)/'all'
    ensure_dir_exists(save_all_to)

    if skip_existing:
        encoded = [gen_enc_filename(fname)
                   for fname in os.listdir(save_all_to)]

    encode_midi_files(
        list(midi_files), save_to, instrument_filter=instrument_filter, no_transpose=False, already_encoded=encoded)

    # Once we've encoded all the files we can, split them up into valid, train and test directories
    encoded = [(Path(dest)/'all')/enc for enc in os.listdir(save_all_to)]
    random.shuffle(encoded)

    test_idx = math.floor(len(enc) * (1 - test))
    valid_idx = math.floor(len(enc) * (1 - test - valid))

    split_files = {
        'train': encoded[:valid_idx],
        'valid': encoded[valid_idx: test_idx],
        'test': encoded[test_idx:]
    }

    for type_, fnames in split_files.items():
        save_to = Path(dest)/type_
        ensure_dir_exists(save_to)
        for fname in fnames:
            copyfile(fname, save_to)


def gen_dataset_from_csv(csv_file, dest, instrument_filter, no_transpose=False):
    df = pd.read_csv(csv_file)
    for type_ in ('train', 'valid', 'test'):
        filtered = df[df['type'] == type_]
        save_to = Path(dest)/type_
        ensure_dir_exists(save_to)
        encode_midi_files(
            list(filtered['file']), save_to, instrument_filter=instrument_filter, no_transpose=False)


def run(args):
    artists = args.get('artists')
    midi_files_csv = args.get('midi_files')
    dest = args.get('dest')
    if artists:
        gen_dataset_from_artists(
            artists, dest,
            instrument_filter=args.get(
                'instrument_filter'),
            no_transpose=args.get('no_transpose', False))
    elif midi_files_csv:
        gen_dataset_from_csv(midi_files_csv, dest,
                             instrument_filter=args.get(
                                 'instrument_filter'),
                             no_transpose=args.get('no_transpose', False))
