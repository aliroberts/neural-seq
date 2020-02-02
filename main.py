
import argparse
from functools import partial
import sys

from src import NeuralSeqUnrecognisedArgException
from src.commands import encode, decode, fetch_data, gen_dataset, improv, list_artists, list_songs, play_midi, train
from src.utils.models import fetch_model
from src.utils.system import get_kwarg_dict
from src.constants import DATA_DIR


def main():
    parser = argparse.ArgumentParser(prog='main')
    subparsers = parser.add_subparsers(dest='subparser')

    # Decoding of string format to MIDI file
    parser_decode = subparsers.add_parser('decode')
    parser_decode.add_argument('--encoder',
                               help='The filename of the encoder to use', type=str, required=True)
    parser_decode.add_argument(
        '--dir', default=None, help='A directory containing the files that should be decoded', type=str)
    parser_decode.add_argument(
        '--file', default=None, help='A file that should be decoded', type=str)
    parser_decode.add_argument('--dest', default=None, required=True,
                               help='The directory to save the decoded MIDI file(s)', type=str)
    parser_decode.add_argument('--tempo', default=120,
                               help='Tempo of decoded MIDI', type=int)
    parser_decode.set_defaults(func=decode.run)

    # Encoding of MIDI files to string format
    parser_encode = subparsers.add_parser('encode')
    parser_encode.add_argument('--encoder',
                               help='The filename of the encoder to use', type=str, required=True)
    parser_encode.add_argument(
        '--dir', default=None, help='A directory containing the MIDI files that should be encoded', type=str)
    parser_encode.add_argument(
        '--file', default=None, help='A MIDI file that should be encoded.', type=str)
    parser_encode.add_argument('--dest', default=None, required=True,
                               help='The directory to save the encodings of the specified MIDI file(s)', type=str)
    parser_encode.add_argument('--filter', default='bass',
                               help='A filter to be applied to the instrument name in the MIDI file', type=str)
    parser_encode.add_argument('--no-transpose', default=False, action='store_true',
                               help='Do not transpose each MIDI file (produce a single example from a MIDI file rather than a collection in various keys)')
    parser_encode.set_defaults(func=encode.run)

    # Data fetching
    parser_fetch = subparsers.add_parser('fetch-data')
    parser_fetch.set_defaults(func=fetch_data.run)

    # Dataset generation
    parser_gen_dataset = subparsers.add_parser('gen-dataset')
    parser_gen_dataset.add_argument(
        '--artists', default=None, help='A text file containing a newline delimited list of artists whose songs should be included in the dataset', type=str)
    parser_gen_dataset.add_argument(
        '--midi-files', default=None, help='A CSV with file, type={train, valid, test} columns to encode and split accordingly', type=str)
    parser_gen_dataset.add_argument('--dest', default=DATA_DIR/'dataset',
                                    help='Where to output the dataset', type=str)
    parser_gen_dataset.add_argument('--encoder',
                                    help='The filename of the encoder to use', type=str, required=True)
    parser_gen_dataset.add_argument('--filter', default='bass',
                                    help='A filter to be applied to the instrument name in the MIDI file', type=str)
    parser_gen_dataset.add_argument('--no-transpose', default=False, action='store_true',
                                    help='Do not transpose each MIDI file (produce a single example from a MIDI file rather than a collection in various keys)')
    parser_gen_dataset.add_argument('--valid', default=0,
                                    help='Proportion of the input files to reserve for the validation set (Default 0)', type=float)
    parser_gen_dataset.add_argument('--test', default=0,
                                    help='Proportion of the input files to reserve for the test set (Default 0)', type=float)
    parser_gen_dataset.set_defaults(func=gen_dataset.run)

    # Music generation
    parser_improv = subparsers.add_parser('improv')
    parser_improv.add_argument('--model', required=True,
                               help='Model type', type=str)
    parser_improv.add_argument('--state', required=True,
                               help='Path to saved model state', type=str)
    parser_improv.add_argument('--encoder', required=True,
                               help='The filename of the encoder to use', type=str)
    parser_improv.add_argument('--rec',
                               help='Path to save the generated MIDI sequence', type=str)
    parser_improv.add_argument('--seq', default=16,
                               help='Length of musical sequence to produce (quarter notes)', type=int)
    parser_improv.add_argument('--prompt', default='',
                               help='A comma-delimited string of encoded tokens', type=str)
    parser_improv.add_argument('--loop', default=4,
                               help='Number of times to loop the generated sequence', type=int)
    parser_improv.add_argument('--tempo', default=120,
                               help='Tempo of generated MIDI', type=int)
    parser_improv.add_argument('--sample', default='topk',
                               help='Sampling strategy for generation [ml|top-<n>|nucleus]', type=str)
    parser_improv.add_argument('--k', default=3,
                               help='Set k for beam or topk sampling', type=int)
    parser_improv.set_defaults(func=improv.run)

    # List artists
    parser_list_artists = subparsers.add_parser('list-artists')
    parser_list_artists.add_argument('--search', default=None,
                                     help='Search for artists containing the specified string', type=str)
    parser_list_artists.set_defaults(func=list_artists.run)

    # List songs
    parser_list_songs = subparsers.add_parser('list-songs')
    parser_list_songs.add_argument('--artist', required=True, default=None,
                                   help='List songs for a given artist (exact match, case sensitive)', type=str)
    parser_list_songs.add_argument('--path', required=False, action='store_true', default=False,
                                   help='List relative paths to the files associated with the specified particular artist')
    parser_list_songs.set_defaults(func=list_songs.run)

    # Play MIDI
    parser_play_midi = subparsers.add_parser('play-midi')
    parser_play_midi.add_argument('midi_file',
                                  help='Play the specified MIDI file from the terminal', type=str)
    parser_play_midi.add_argument('--filter', default=None,
                                  help='A filter to be applied to the instrument name in the MIDI file', type=str)
    parser_play_midi.add_argument('--tempo', default=120,
                                  help='Tempo for playback', type=int)
    parser_play_midi.set_defaults(func=play_midi.run)

    # Training
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model', required=True, type=str)
    parser_train.add_argument('--data-dir', required=True)
    parser_train.add_argument('--dest', required=True,
                              help='Directory in which to save model snapshots')
    parser_train.add_argument('--epochs', default=10,
                              help='Epochs to train model for (default 10)', type=int),
    parser_train.add_argument('--bs', default=32, help='Batch size', type=int)
    parser_train.add_argument('--bptt', default=128, help='BPTT', type=int)

    model_kwargs = {}
    train_kwargs = {}

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        model_args = parser_train.parse_known_args()
        _, train_func, model_kwargs = fetch_model(model_args[0].model)

        train_kwargs = get_kwarg_dict(train_func)

        parser_train.add_argument(
            '--pretrained', default=None, help='Model with which to start training', type=str)

        # Add any model kwargs and train function kwargs as command line options
        for arg, default in model_kwargs.items():
            parser_train.add_argument(
                f'--{arg}', default=default, type=type(default))

        for arg, default in train_kwargs.items():
            parser_train.add_argument(
                f'--{arg}', default=default, type=type(default))

    parser_train.add_argument(
        '--save-freq', default=0, help='Frequency at which to save model snapshots (every SAVE_FREQ epochs)', type=int)
    parser_train.add_argument(
        '--prefix', default='model', help='Prefix for saving model (default mod)')
    parser_train.add_argument(
        '--cache', default=False, action='store_true', help='Update the cached data bunch (use when updating the batch size!)')
    parser_train.add_argument(
        '--resume', default=False, action='store_true', help='Resume from the latest epoch (determined by model name <prefix>-epoch-<epoch>)')

    parser_train.set_defaults(func=partial(
        train.run, model_kwargs=model_kwargs, train_kwargs=train_kwargs))

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    try:
        args = parser.parse_args()
        args.func(args)
    except NeuralSeqUnrecognisedArgException:
        parser.print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
