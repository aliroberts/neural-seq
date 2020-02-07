
import io
import pretty_midi

from src import NeuralSeqEncodingException
from src.utils.midi_encode import BaseEncoder


def round_to(val, rnd):
    return (val // rnd) * rnd


class DrumEncoder(BaseEncoder):
    SMALLEST_DIVISION = 8

    def encode(self, midi):
        _, tempi = midi.get_tempo_changes()
        tempo = tempi[0]

        drums = None

        for inst in midi.instruments:
            if inst.is_drum:
                drums = inst

        if not drums:
            raise NeuralSeqEncodingException('Instrument was not found')

        resolution = 60 / tempo / self.SMALLEST_DIVISION

        drums.notes = sorted(drums.notes, key=lambda x: x.start)

        toks = []
        prev_start = 0
        durations = []

        for note in drums.notes:
            # Snap to nearest time step
            start = round_to(note.start, resolution)
            if start > 0 and start == prev_start:
                toks.append(f'P-{note.pitch}')
            else:
                # Compress any rests longer than one bar (assuming 4/4)
                duration = int((start - prev_start) //
                               resolution) % self.SMALLEST_DIVISION * 4
                toks.append(f'D-{duration}')
                toks.append(f'P-{note.pitch}')
                prev_start = start

        if len(toks) < 32:
            raise NeuralSeqEncodingException('Sequence length too short')
        return toks

    def decode(self, enc, midi_programs=None, tempo=None):
        decoded = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drums = pretty_midi.Instrument(0, is_drum=True)

        resolution = 60 / tempo / self.SMALLEST_DIVISION

        notes = drums.notes
        prev_note = None
        prev_tok = None
        dur = 0

        for tok in enc:
            if tok[0] == 'P':
                # Handle play action
                start = dur if not prev_note else prev_note.start + dur
                end = start + resolution
                vel = 100
                pitch = int(tok.replace('P-', ''))
                note = pretty_midi.Note(
                    velocity=vel, pitch=pitch, start=start, end=start+resolution)

                notes.append(note)
                prev_note = note

                # Reset duration counter
                dur = 0
            else:
                # Keep a running total of how long we should wait before starting the next action
                # we come across
                dur += int(tok.replace('D-', '')) * resolution

        decoded.instruments.append(drums)
        return decoded

    def process_prediction(self, prediction, seq_length=None):
        # If a specified sequence length has been provided, truncate
        # the prediction to fit (the number of tokens is not the same
        # as the duration/sequence length of the provided encodings)
        print(prediction)
        processed = []
        duration = 0

        for tok in prediction:
            if tok[0] == 'D':
                tok_dur = int(tok.replace('D-', ''))

                # Truncate the duration if we will go over the seq_length limit
                remaining = seq_length - (duration + tok_dur)

                if remaining < 0:
                    duration += tok_dur + remaining
                else:
                    duration += tok_dur
                tok = f'D-{tok_dur}'
            processed.append(tok)
        return processed

    def duration(self, enc):
        return sum([int(tok.replace('D-', '')) for tok in enc if 'D-' in tok])
