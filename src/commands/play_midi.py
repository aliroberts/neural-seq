
import music21


def run(args):
    with open(args.midi_file, 'rb') as f:
        mf = music21.midi.MidiFile()
        mf.openFileLike(f)
        mf.read()
    stream = music21.midi.translate.midiFileToStream(mf)

    if args.instrument_filter:
        stream = music21.instrument.partitionByInstrument(stream)
        for p in stream.parts:
            if p.partName and args.instrument_filter in p.partName.lower():
                # See https://web.mit.edu/music21/doc/moduleReference/moduleInstrument.html for more info
                p.makeRests(inPlace=True, hideRests=True)
                p.makeMeasures(inPlace=True)
                p.makeTies(inPlace=True)
                stream = p
                break
    sp = music21.midi.realtime.StreamPlayer(stream)
    sp.play()
