import math

import music21

from src.utils.midi_encode import BaseEncoder
from src.encoders.shr_mono import StartHoldRestEncoder


class NotewiseMonoEncoder(BaseEncoder):
    @staticmethod
    def process_duration_buffer(bf):
        # Express duration values as powers of 2
        bin_str = bin(len(bf))[2:]
        processed = []
        for i, char in enumerate(list(reversed(bin_str))):
            if char == '1':
                processed.append(f'D-{2**i}')
        return list(reversed(processed)), len(bf)

    def encode(self, stream, sample_freq=4, transpose=0):
        # Monophonic encoding
        # A slightly modified version of Christine Mcleavey's Clara encoder
        # https://github.com/mcleavey/musical-neural-net/tree/master/data

        # Describes the musical sequence a combination of actions and durations
        # E.g. S-41,D-8,R,D-2,S-44,D-2,D-4 would describe the following actions:
        #   1. Initiate MIDI note 41, hold for 8 of whatever note duration the sample_freq represents (semi-quavers by default)
        #   2. Terminate note (Rest), hold for 2 semi-quavers
        #   3. Initiate MIDI note 44, hold for 2 + 4 = 6 of whatever note duration the sample_freq represents (semi-quavers by default)

        shr_encoder = StartHoldRestEncoder()
        shr_encoded = shr_encoder.encode(stream, transpose=transpose)

        duration = 0
        encoded = []
        duration_buffer = []
        for action in shr_encoded:

            if action[0] == 'H':
                duration_buffer.append('D-1')
            elif action == 'R' and (encoded and encoded[-1] == 'R'):
                duration_buffer.append('D-1')
            else:
                comp, dur = self.process_duration_buffer(duration_buffer)
                duration += dur
                encoded += comp
                encoded.append(action)
                duration_buffer = ['D-1']
        encoded += self.process_duration_buffer(duration_buffer)
        return encoded

    def decode(self, enc_notes, sample_freq=4, seq_length=None):
        if isinstance(enc_notes, str):
            enc_notes = enc_notes.split(',')
        duration_inc = 1 / sample_freq

        notes = []

        duration_acc = 0
        curr_note = None
        for enc in enc_notes:
            if enc[0] == 'S':
                curr_note = music21.note.Note()
                curr_note.pitch.midi = int(enc.replace('S-', ''))
                curr_note.duration = music21.duration.Duration(0)
                notes.append(curr_note)
            elif enc[0] == 'R':
                curr_note = music21.note.Rest()
                curr_note.duration = music21.duration.Duration(0)
                notes.append(curr_note)
            else:
                curr_note.duration.quarterLength += duration_inc * \
                    int(enc.replace('D-', ''))
        return notes

    def process_prediction(self, prediction, seq_length=None):
        # If a specified sequence length has been provided, truncate
        # the prediction to fit (the number of tokens is not the same
        # as the duration/sequence length of the provided encodings)
        print(prediction)
        processed = []
        curr_seq_len = 0
        for tok in prediction:
            if seq_length and tok[0] == 'D':
                dur = int(tok.replace('D-', ''))
                remaining = seq_length - curr_seq_len

                if dur > remaining:
                    processed += self.process_duration_buffer(
                        ['D-1' for _ in range(remaining)])[0]
                    break
                else:
                    processed.append(tok)
                    curr_seq_len += dur
            else:
                prev_is_action = len(
                    processed) > 0 and processed[-1] in ('S', 'R')
                if prev_is_action and tok[0] in ('S', 'R'):
                    continue

                processed.append(tok)
        print(processed)
        return processed

    def duration(self, encoded):
        return sum([int(tok.replace('D-', '')) for tok in encoded if 'D-' in tok])
