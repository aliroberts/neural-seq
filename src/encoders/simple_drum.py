
import io
import pretty_midi

from src.utils.midi_encode import BaseEncoder


def round_to(val, rnd):
    return (val // rnd) * rnd


class DrumEncoder(BaseEncoder):
    def encode(self, midi):
        _, tempi = midi.get_tempo_changes()
        tempo = tempi[0]

        drums = None

        for inst in midi.instruments:
            if inst.is_drum:
                drums = inst

        resolution = 60 / tempo / 4  # 16th notes

        toks = []
        prev_start = 0
        durations = []

        for note in drums.notes:
            # Snap to nearest time step
            start = round_to(note.start, resolution)
            if start > 0 and start == prev_start:
                toks.append(f'P-{note.pitch}')
            else:
                duration = int((start - prev_start) // resolution)
                toks.append(f'D-{duration}')
                toks.append(f'P-{note.pitch}')
                prev_start = start
        return toks

    def decode(self, enc, midi_programs, tempo=None):
        decoded = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drums = pretty_midi.Instrument(0, is_drum=True)

        resolution = 60 / tempo / 4  # 16th notes

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

        for tok in enc:
            if tok[0] == 'P':
                if prev_tok and prev_tok[0] == 'P':
                    # The previous token was a play action, so we want to create a chord (start this one
                    # at the same time)
                    start = notes[-1].start
                elif prev_tok and prev_tok[0] == 'D':
                    # The previous token was a duration, so we want to start the new note at the time of the
                    # previous one's start + length of the duration in secs
                    if prev_note:
                        prev_start = prev_note.start
                    else:
                        prev_start = 0
                    start = prev_start + \
                        int(prev_tok.replace('D-', '')) * resolution
                else:
                    # No previous token, start from the beginnning
                    start = 0
                vel = 100
                pitch = int(tok.replace('P-', ''))
                note = pretty_midi.Note(
                    velocity=vel, pitch=pitch, start=start, end=start+resolution)
                notes.append(note)
                prev_note = note
            elif prev_note:
                dur += int(tok.replace('D-', '')) * resolution

            prev_tok = tok
        decoded.instruments.append(drums)
        return decoded
