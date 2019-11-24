import os
from pprint import pprint

from src.utils.midi_data import midi_data_required

from src.constants import MIDI_ARTISTS


@midi_data_required
def run(args):
    all_artists = filter(
        lambda s: args.search.lower() in s.lower() if args.search else s, os.listdir(MIDI_ARTISTS))
    if not all_artists:
        print('NO ARTISTS FOUND')
    else:
        print('\n'.join(all_artists))
