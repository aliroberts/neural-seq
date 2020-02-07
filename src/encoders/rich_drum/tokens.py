import json
import os
from pathlib import Path

from src.constants import MIDI_DRUM_PATCH_NUMS

file_loc = Path(os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__))))

TOP_TOKS = Path(file_loc/'tokens.json')


def process_midi_program(midi_prog):
    """
    Map instruments to some canonical version (all bass drums to the 'Acoustic Bass Drum'
    program for example)
    """
    replace = {
        36: 35,
        40: 38,
        51: 49,
        57: 51
    }
    return replace.get(midi_prog, midi_prog)


def load_top_toks():
    with open(TOP_TOKS) as f:
        tok_data = json.load(f)

    toks = []
    for item in tok_data:
        dur = item.pop()
        program_nums = list(
            map(lambda x: f'P-{MIDI_DRUM_PATCH_NUMS[x]}', item))
        tok = f'{("_").join(program_nums)}_D-{dur}'
        toks.append(tok)
    return set(toks)
