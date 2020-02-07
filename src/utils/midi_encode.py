import math
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import music21
import pretty_midi

from src import NeuralSeqEncodingException
from src.constants import ENCODER_DIR, MIDI_PATCH_NAMES
from src.utils.system import fetch_subclass_from_file


def gen_enc_filename(fpath):
    return os.path.basename(fpath).split('.')[0]


class BaseEncoder(object):
    def encode(self, midi, sample_freq=4, transpose=0):
        """
        Return a list-of-strings representation of the notes for downstream tasks (language modeling).
        Takes a PrettyMIDI instance.
        """
        raise NotImplementedError

    def decode(self, encoded, midi_programs=None, tempo=None):
        """
        Return a MIDI file containing the decoding for the specified encoding. midi_programs contains a list
        of the programs (corresponding to instruments) that were included in the original pre-encoded MIDI file.
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
        raise NotImplementedError


def fetch_encoder(name):
    """
    Provide the filename of the encoder and locate it in the ENCODER_DIR. Return the encoder class.
    """
    return fetch_subclass_from_file(ENCODER_DIR, name.replace('.py', ''), BaseEncoder, strict=True)


class MIDIData(object):
    """
    A thin wrapper around pretty_midi's PrettyMIDI class which wraps it up with and encoder.
    """
    @classmethod
    def from_file(cls, fname, encoder, instrument_filter=None):
        """
        Load a MIDIData instance from the specified MIDI file and with the supplied encoder instance.
        """
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(str(fname))
        for inst in midi_data.instruments:
            try:
                name = MIDI_PATCH_NAMES[inst.program] if not inst.is_drum else 'Drum Kit'
            except IndexError:
                continue

            if instrument_filter(name.lower()):
                print(
                    f'Found instrument matching filter: {name}')
                midi_data.instruments = [inst]
                break
        else:
            print(f'Warning: No instrument matching filter')
            midi_data.instruments = []
        return cls(midi_data, encoder)

    @classmethod
    def encode_files(cls, files, dest, encoder, prefix='', instrument_filter='',
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
                midi_data = cls.from_file(
                    fpath, encoder, instrument_filter=instrument_filter)
            except Exception as e:
                print(instrument_filter)
                print('Unable to load file, skipping...')
                continue
            print(
                f'DONE (found {len(midi_data)} tracks matching filter)')

            print(
                f'----> Encoding...')
            try:
                if no_transpose:
                    encoded = [midi_data.encode()]
                else:
                    encoded = midi_data.encode_range()
            except NeuralSeqEncodingException:
                print('Encoding failed. Skipping...')
                continue

            if ignore_duplicates:
                ignore.append(gen_enc_filename(fpath))

            print(f'----> Writing files ({len(encoded)}) to {dest}')
            for i, enc in enumerate(encoded):
                vocab = vocab.union(set(enc))
                _, tempi = midi_data.midi_data.get_tempo_changes()

                enc_data = {
                    'data': enc,
                    'vocab': list(vocab),
                    'tempo': tempi[0],
                    'programs': [int(inst.program) for inst in midi_data.midi_data.instruments]
                }
                # NOTE: We save the files with a txt extension so FastAI's TextLMDataBunch can be used easily
                with open(Path(dest)/f'{out_name}-{i+1}.txt', 'w') as f:
                    json.dump(enc_data, f)
            print('----> DONE')
        return list(vocab)

    @classmethod
    def decode_files(cls, files, dest, encoder, tempo=None):
        """
        Takes a list of file names, an output directory and an encoder and decodes the encoded files into
        midi files. If a tempo has not been provided then use the one in the file containing encoding data
        if there is one.
        """

        for fpath in files:
            with open(fpath) as f:
                enc_data = json.load(f)
            print(f'Decoding {fpath}')
            enc, midi_programs, enc_tempo = enc_data['data'], enc_data['programs'], enc_data['tempo']
            tempo = tempo if tempo else enc_tempo
            decoded = encoder.decode(
                enc, midi_programs=midi_programs, tempo=tempo)
            out_fname = os.path.splitext(fpath)[0] + '.mid'
            out_path = Path(dest)/os.path.basename(out_fname)
            print(f'----> Writing file to {out_path}')
            with open(out_path, 'wb') as f:
                decoded.write(f)
            print('----> DONE')

    def __init__(self, midi_data, encoder):
        self.midi_data = midi_data
        self.encoder = encoder

    def encode(self):
        """
        Create an encoding for the instance's midi_data, transposing the pitches accordingly.
        NOTE: Will not attempt to transpose the drum kit (if there is one)
        """
        return self.encoder.encode(self.midi_data)

    def encode_range(self):
        """
        Create a batch of encodings for a single MIDIData object that are transposed into as many keys as possible
        (given the range of the instruments)
        """
        return [self.encoder.encode(self.midi_data)]

    def __len__(self):
        return len(self.midi_data.instruments)


def encode_midi_files(files, dest, encoder, prefix='', instrument_filter='', midi_range=(24, 75),
                      transpose=False, ignore_duplicates=True, already_encoded=None, gen_enc_filename=gen_enc_filename):
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
                if not transpose:
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
