import math
from collections import defaultdict
import os
from pathlib import Path
import sys

import music21

from src.constants import ENCODER_DIR
from src.utils.system import fetch_subclass_from_file


class BaseEncoder(object):
    def encode(self, stream, sample_freq=4, transpose=0):
        """
        Return a list-of-strings representation of the notes for downstream tasks (language modeling)
        """
        raise NotImplementedError

    def decode(self, enc_notes, sample_freq=4):
        """
        Return a list of music21.note.Note/Rest instances from a list-of-strings representation sampled at the
        specified sample frequency.
        """
        raise NotImplementedError

    def process_prediction(self, prediction, seq_length=None):
        """
        Perform any post-processing of a raw prediction from a model before it is passed to the decoder
        including any truncating of the sequence so it's the specified length.
        """
        return prediction

    def duration(self, encoded):
        """
        Takes a sequence of encoded tokens and calculates the musical duration of the encoded sequence
        (the number returns the number of 'beats' of sample frequency duration with which the encoding was made)
        """
        return NotImplementedError


def fetch_encoder(name):
    """
    Provide the filename of the encoder and locate it in the ENCODER_DIR. Return the encoder class.
    """
    return fetch_subclass_from_file(ENCODER_DIR, name.replace('.py', ''), BaseEncoder, strict=True)


class Track(object):
    def __init__(self, name, stream, encoder):
        self.name = name
        self.stream = stream  # music21.stream.Stream()
        self.encoder = encoder

    def encode(self, transpose=0):
        return self.encoder.encode(self.stream, transpose=transpose)

    def track_range(self):
        pitches = [note.pitch.midi for note in self.stream.notes if isinstance(
            note, music21.note.Note)]
        return (min(pitches), max(pitches))

    def encode_many(self, rng=(24, 75)):
        """
        Returns a list of encodings that fit between the provided range and preserve the original
        note relationships (if any can).
        """
        rng_min, rng_max = rng
        pitch_min, pitch_max = self.track_range()

        trans_min = rng_min - pitch_min
        trans_max = trans_min + (rng_max - pitch_max)

        encodings = []

        if (rng_max - rng_min) < (pitch_max - pitch_min):
            return encodings

        for transpose in range(trans_min, trans_max + 1):
            encodings.append(self.encode(transpose=transpose))
        return encodings


class TrackCollection(object):
    """
    A wrapper around music21's Parts that contains some loading helpers and
    wraps the parts in Track classes that have additional encoding/decoding
    functionality
    """

    def __init__(self, tracks):
        self.tracks = tracks

    @classmethod
    def load(cls, stream, encoder, instrument_filter=None):
        """
        Load a track collection with a collection of parts for each instrument
        """
        stream = music21.instrument.partitionByInstrument(stream)
        tracks = []
        for p in stream.parts:
            if instrument_filter and not(p.partName and instrument_filter(p.partName.lower())):
                continue

            # Assigns actions to measures and fills in Rests
            # See https://web.mit.edu/music21/doc/moduleReference/moduleInstrument.html for more info
            # p.makeRests(inPlace=True, hideRests=True)
            # p.makeMeasures(inPlace=True)
            # p.makeTies(inPlace=True)
            tracks.append(Track(p.partName, p, encoder))
        return cls(tracks)

    @classmethod
    def load_from_file(cls, fname, encoder, instrument_filter=None):
        with open(fname, 'rb') as f:
            mf = music21.midi.MidiFile()
            mf.openFileLike(f)
            mf.read()
        stream = music21.midi.translate.midiFileToStream(mf)
        return cls.load(stream, encoder, instrument_filter=instrument_filter)


def gen_enc_filename(fpath):
    return os.path.basename(fpath).split('.')[0]


def encode_midi_files(files, dest, encoder, prefix='', instrument_filter='', midi_range=(24, 75),
                      no_transpose=False, ignore_duplicates=True, already_encoded=None, gen_enc_filename=gen_enc_filename):
    """
    Takes a list of midi file paths and an output destination in which to save the encodings.
    """
    # Keep track of the files that we've successfully encoded so that we don't end up encoding a duplicate
    # file (there are several different versions of the the same track in the cleaned MIDI dataset, for example)
    ignore = already_encoded.copy() if already_encoded else []
    vocab = set()
    for fpath in files:
        print(f'Processing {fpath}...')
        if gen_enc_filename(fpath) in ignore:
            print('Duplicate detected. Skipping...')
            continue
        out_name = Path(os.path.basename(fpath)).with_suffix('')
        try:
            track_collection = TrackCollection.load_from_file(
                fpath, encoder, instrument_filter=instrument_filter)
        except Exception as e:
            print(instrument_filter)
            raise
            print('Unable to load file, skipping...')
            continue
        print(
            f'DONE (found {len(track_collection.tracks)} tracks matching filter)')
        for track in track_collection.tracks:
            print(f'----> Encoding {track.name}')
            try:
                if no_transpose:
                    encoded = [track.encode()]
                else:
                    encoded = track.encode_many(midi_range)
            except ValueError:
                raise
                print('Encoding failed. Skipping...')
                continue

            if ignore_duplicates:
                ignore.append(gen_enc_filename(fpath))

            print(f'----> Writing files ({len(encoded)}) to {dest}')
            for i, enc in enumerate(encoded):
                vocab = vocab.union(set(enc))
                with open(Path(dest)/f'{out_name}-{i+1}.txt', 'w') as f:
                    f.write(','.join(enc))
            print('----> DONE')
    return list(vocab)


def decode_files(files, dest, encoder, tempo=120):
    for fpath in files:
        with open(fpath) as f:
            enc_seq = f.read()
        print(f'Decoding {fpath}')
        dec_notes = encoder.decode(enc_seq)
        score = music21.stream.Score()
        dec_stream = music21.stream.Part()
        inst = music21.instrument.Piano()
        dec_stream.insert(0, inst)
        score.insert(0, dec_stream)
        t = music21.tempo.MetronomeMark(number=tempo)
        dec_stream.append(t)
        for note in dec_notes:
            dec_stream.append(note)
        inst.bestName()
        out_fname = os.path.splitext(fpath)[0] + '.mid'
        out_path = Path(dest)/os.path.basename(out_fname)
        print(f'----> Writing file to {out_path}')
        mf = music21.midi.translate.streamToMidiFile(score)
        mf.open(out_path, 'wb')
        mf.write()
        mf.close()
        print('----> DONE')
