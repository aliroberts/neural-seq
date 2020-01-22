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
        return list(reversed(processed))

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

        encoded = []
        duration_buffer = []
        for action in shr_encoded:
            if action[0] == 'H':
                duration_buffer.append('D-1')
            elif action == 'R' and (encoded and encoded[-1] == 'R'):
                duration_buffer.append('D-1')
            else:
                encoded += self.process_duration_buffer(duration_buffer)
                encoded.append(action)
                duration_buffer = ['D-1']
        encoded += self.process_duration_buffer(duration_buffer)
        return encoded

    def decode(self, enc_notes, sample_freq=4):
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

    def process_prediction(self, prediction):
        print(prediction)
        tokens = prediction
        prev_tok = None

        cleaned = []
        for t in tokens:
            if prev_tok and prev_tok[0] == 'H' and t[0] == 'H':
                if prev_tok.split('-')[1] != t.split('-')[1]:
                    t = t.replace('H', 'S')
            cleaned.append(t)
            prev_tok = t
        return cleaned
