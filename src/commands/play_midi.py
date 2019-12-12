
import music21


def run(args):
    with open(args.midi_file, 'rb') as f:
        mf = music21.midi.MidiFile()
        mf.openFileLike(f)
        mf.read()
    stream = music21.midi.translate.midiFileToStream(mf)

    if args.instrument_filter:
        note_seq = []
        instrument_stream = music21.stream.Stream()
        def instrument_filter(
            instrument): return args.instrument_filter in str(instrument).lower()

        loading_instrument = False
        for action in stream.recurse(classFilter=('Note', 'Rest')):
            instrument = action.activeSite.getInstrument()
            if instrument_filter and not instrument_filter(instrument):
                if loading_instrument:
                    break
                else:
                    continue
            loading_instrument = True
            instrument_stream.append(action)
        stream = instrument_stream

    sp = music21.midi.realtime.StreamPlayer(stream)
    sp.play()
