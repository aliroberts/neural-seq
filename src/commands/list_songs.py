import os
from pprint import pprint

from src.utils.midi_data import midi_data_required

from src.constants import MIDI_ARTISTS


@midi_data_required
def run(args):
    artist = args.artist

    if artist not in os.listdir(MIDI_ARTISTS):
        print('ARTIST NOT FOUND')
        return

    try:
        songs = os.listdir(MIDI_ARTISTS/artist)
    except FileNotFoundError:
        print('NO SONGS FOUND')

    if args.path:
        print('\n'.join([str(MIDI_ARTISTS/artist/song) for song in songs]))
    else:
        print('\n'.join(songs))
