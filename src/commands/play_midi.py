import io
import os
import pretty_midi
import sys

from src.constants import MIDI_PATCH_NAMES


def run(args):
    midi_data = pretty_midi.pretty_midi.PrettyMIDI(args.midi_file)
    if args.filter:
        filter_name = args.filter.lower()
        for inst in midi_data.instruments:
            try:
                name = MIDI_PATCH_NAMES[inst.program] if not inst.is_drum else 'Drum Kit'
            except IndexError:
                continue

            if filter_name in name.lower():
                print(
                    f'Found instrument matching filter \'{args.filter}\': {name}')
                midi_data.instruments = [inst]
                break
        else:
            print(f'No instrument matching filter \'{args.filter}\'')
            sys.exit(0)

    buff = io.BytesIO()
    midi_data.write(buff)
    buff.seek(0)

    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(buff)
    try:
        pygame.mixer.music.play()
        print(f'Playing {args.midi_file} (Ctrl+C to stop)')
        while pygame.mixer.music.get_busy():
            pygame.time.wait(1000)
    except KeyboardInterrupt:
        sys.exit(0)
