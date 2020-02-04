import io
import os
import sys
import tarfile

from src.utils.download import download_file
from src.utils.system import copyfile, make_temp_dir, yn

from src.constants import MIDI_DATA_URL, MIDI_DIR, MIDI_ARTISTS


def play_midi(midi_data):
    """
    Play a pretty_midi.MidiData object using pygame
    """
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
    import pygame
    buff = io.BytesIO()
    midi_data.write(buff)
    buff.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(buff)
    try:
        pygame.mixer.music.play()
        print(f'Playing... (Ctrl+C to stop)')
        while pygame.mixer.music.get_busy():
            pygame.time.wait(1000)
    except KeyboardInterrupt:
        sys.exit(0)


def fetch_midi_data():
    with make_temp_dir('nn_seq') as dir:
        tar_dest = os.path.join(dir, 'midi_data.tar.gz')
        download_file(MIDI_DATA_URL, tar_dest, 'Downloading MIDI data...')
        print('Extracting...')
        tf = tarfile.open(tar_dest, 'r:gz')
        tf.extractall(MIDI_DIR)
        print('DONE')


def midi_data_exists():
    try:
        files = os.listdir(MIDI_ARTISTS)
    except FileNotFoundError:
        return False
    return len(files) > 10


def midi_data_required(func):
    def wrapper(*args, **kwargs):
        if not midi_data_exists():
            cont = yn(
                'This action requires some MIDI files (~811MB). Download? [y/n] ')
            if not cont:
                sys.exit(1)
        func(*args, **kwargs)
    return wrapper
