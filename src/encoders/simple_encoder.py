from src.utils.midi_encode import BaseEncoder


class SimpleEncoder(BaseEncoder):
    def encode(self, sample_freq, transpose):
        # Monophonic encoding
        # Iterate through the notes/chords/rests. When we encounter a note, add it to the list of encoded notes
        # along with its duration. When we encounter a new event a few things might happen:

        # i) The event occurs before the end of the previous event in which case we want to stop the previous event
        # and trigger the new one.
        # ii) The event occurs after the previous event has finished by some non-zero offset, in which case
        # let's fill the gap with rests before we trigger the new one.
        # iii) The event occurs exactly as the previous one finishes in which case let's simply trigger the new event.

        # Iterate through the notes and snap them to our array.

        # Default array with rest values
        encoded = ['R' for _ in range(int(stream.duration.quarterLength * 4))]

        for note in stream.notes:
            offset = math.floor(note.offset * sample_freq)
            duration = math.floor(note.duration.quarterLength * sample_freq)

            if isinstance(note, music21.chord.Chord):
                note = note.notes[0]

            midi_pitch = note.pitch.midi + transpose

            # Start note
            tok = f'S-{midi_pitch}'
            encoded[offset] = tok

            # Hold note
            for i in range(duration - 1):
                encoded[offset + i + 1] = f'H-{midi_pitch}'
        return encoded

    def decode(self, sample_freq):
        if isinstance(enc_notes, str):
            enc_notes = enc_notes.split(',')
        notes = []
        curr_note = None
        duration_inc = 1 / sample_freq

        for enc in enc_notes:
            if 'S-' in enc:
                curr_note = music21.note.Note()
                notes.append(curr_note)
                curr_note.pitch.midi = int(enc.replace('S-', ''))
                curr_note.duration = music21.duration.Duration(
                    duration_inc)
            elif 'H-' in enc:
                curr_note.duration.quarterLength += duration_inc
            elif 'R' in enc:
                curr_note = music21.note.Rest()
                notes.append(curr_note)
                curr_note.duration = music21.duration.Duration(
                    duration_inc)
        return notes
