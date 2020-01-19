import math
import os
import random
from pathlib import Path
import shutil
import sys

import pandas as pd

from src import NeuralSeqUnrecognisedArgException

from src.utils.midi_data import midi_data_required
from src.utils.midi_encode import encode_midi_files, fetch_encoder, gen_enc_filename,
from src.utils.system import copyfile, dir_names, ensure_dir_exists, yn
from src.constants import MIDI_ARTISTS


@midi_data_required
def gen_dataset_from_artists(artists, dest, encoder, instrument_filter, no_transpose=False,
                             valid=0, test=0, skip_existing=True):
    midi_files = []
    for artist in artists:
        song_dir = MIDI_ARTISTS/artist
        midi_files += [song_dir/song for song in os.listdir(song_dir)]

    # Encode all the midi files to an 'all' directory
    save_all_to = Path(dest)/'all'
    ensure_dir_exists(save_all_to)

    if skip_existing:
        encoded = [gen_enc_filename(fname)
                   for fname in os.listdir(save_all_to)]

    vocab = encode_midi_files(
        list(midi_files), save_all_to, encoder, instrument_filter=instrument_filter, no_transpose=no_transpose, already_encoded=encoded)

    # Once we've encoded all the files we can, split them up into valid, train and test directories
    encoded = [(Path(dest)/'all')/enc for enc in os.listdir(save_all_to)]
    random.shuffle(encoded)

    test_idx = math.floor(len(encoded) * (1 - test))
    valid_idx = math.floor(len(encoded) * (1 - test - valid))

    split_files = {
        'train': encoded[:valid_idx],
        'valid': encoded[valid_idx: test_idx],
        'test': encoded[test_idx:]
    }

    for type_, fnames in split_files.items():
        save_to = Path(dest)/type_
        if fnames:
            ensure_dir_exists(save_to)
        for fname in fnames:
            copyfile(fname, save_to)

    with open(Path(dest)/'vocab', 'w') as f:
        f.write(','.join(vocab))


def gen_dataset_from_csv(csv_file, dest, encoder, instrument_filter, no_transpose=False):
    df = pd.read_csv(csv_file)
    vocab = set()
    for type_ in ('train', 'valid', 'test'):
        filtered = df[df['type'] == type_]
        save_to = Path(dest)/type_
        ensure_dir_exists(save_to)
        set_vocab = encode_midi_files(
            list(filtered['file']), save_to, encoder, instrument_filter=instrument_filter, no_transpose=no_transpose)
        vocab = vocab.union(set_vocab)
    with open(Path(dest)/'vocab', 'w') as f:
        f.write(','.join(vocab))


def run(args):
    artists_file = args.artists
    midi_files_csv = args.midi_files
    dest = args.dest

    encoder = fetch_encoder(args.encoder)

    def instrument_filter(
        instrument): return args.instrument_filter in str(instrument).lower()

    if os.path.isdir(dest):
        should_delete_dir = yn(
            f'The destination directory {dest} already exists, would you like to delete it? If you do not delete it, new files will be added to it and any old ones will be kept. [y/n] ')
        if should_delete_dir:
            try:
                shutil.rmtree(dest)
            except OSError as e:
                print(f'An error occurred: {e.filename} - {e.strerror}.')
                sys.exit(1)

    if artists_file:
        with open(artists_file) as f:
            artists = [artist.replace('\n', '') for artist in f.readlines()]
        gen_dataset_from_artists(
            artists, dest, encoder,
            instrument_filter,
            no_transpose=args.no_transpose, valid=args.valid, test=args.test)
    elif midi_files_csv:
        gen_dataset_from_csv(midi_files_csv, dest, encoder,
                             instrument_filter=ainstrument_filter,
                             no_transpose=args.no_transpose,)
    else:
        raise NeuralSeqUnrecognisedArgException
