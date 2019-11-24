
import argparse
import sys

from src import NeuralSeqUnrecognisedArgException
from src.commands import encode, decode, fetch_data, gen_dataset, list_artists, list_songs, train

from src.constants import DATA_DIR


def main():
    parser = argparse.ArgumentParser(prog='main')
    subparsers = parser.add_subparsers(dest='subparser')

    # Decoding of string format to MIDI file
    parser_decode = subparsers.add_parser('decode')
    parser_decode.set_defaults(func=decode.run)

    # Encoding of MIDI files to string format
    parser_encode = subparsers.add_parser('encode')
    parser_encode.add_argument(
        '--files', default=None, help='A text file containing a newline delimited list of files to encode.', type=str)
    parser_encode.add_argument(
        '--dirs', default=None, help='A text file containing a newline delimited list of directories containing files to encode.', type=str)
    parser_encode.add_argument('--dest', default=None, required=True,
                               help='A text file containing a newline delimited list of directories containing files to encode.', type=str)
    parser_encode.set_defaults(func=encode.run)

    # Data fetching
    parser_fetch = subparsers.add_parser('fetch-data')
    parser_fetch.set_defaults(func=fetch_data.run)

    # Dataset generation
    parser_gen_dataset = subparsers.add_parser('gen-dataset')
    parser_gen_dataset.add_argument(
        '--artists', default=None, help='A text file containing a newline delimited list of artists whose songs should be included in the dataset.', type=str)
    parser_gen_dataset.add_argument(
        '--midi-files', default=None, help='A CSV with file, type={train, valid, test} columns to encode and split accordingly', type=str)
    parser_gen_dataset.add_argument('--dest', default=DATA_DIR/'dataset',
                                    help='Where to output the dataset', type=str)
    parser_gen_dataset.add_argument('--instrument-filter', default='bass',
                                    help='A filter to be applied to the instrument name in the MIDI file', type=str)
    parser_gen_dataset.add_argument('--no-transpose', default=False, action='store_false',
                                    help='Do not transpose each MIDI file (produce a single example from a MIDI file rather than a collection in various keys)')
    parser_gen_dataset.set_defaults(func=gen_dataset.run)

    # List artists
    parser_list_artists = subparsers.add_parser('list-artists')
    parser_list_artists.add_argument('--search', default=None,
                                     help='Search for artists containing the specified string', type=str)
    parser_list_artists.set_defaults(func=list_artists.run)

    # List songs
    parser_list_songs = subparsers.add_parser('list-songs')
    parser_list_songs.add_argument('--artist', required=True, default=None,
                                   help='List songs for a given artist (exact match, case sensitive)', type=str)
    parser_list_songs.set_defaults(func=list_songs.run)

    # Training
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--data-dir', required=True)
    parser_train.add_argument('--dropout', default=1,
                              help='Dropout multiplier', type=float)
    parser_train.add_argument(
        '--pre-trained', help='Model with which to start training')
    parser_train.add_argument('--bs', default=32, help='Batch size', type=int)
    parser_train.add_argument('--bptt', default=200, help='BPTT', type=int)
    parser_train.add_argument('--em_sz', default=300,
                              help='Embedding size', type=int)
    parser_train.add_argument(
        '--nh', default=600, help='Number of hidden activations', type=int)
    parser_train.add_argument(
        '--nl', default=4, help='Number of LSTM layers', type=int)
    parser_train.add_argument(
        '--min_freq', default=1, help='Minimum frequency of token for it to be included', type=int)
    parser_train.add_argument('--epochs', default=10,
                              help='Epochs to train model for', type=int),
    parser_train.add_argument(
        '--save_freq', default=1, help='Frequency at which to save model snapshots (every SAVE_FREQ epochs)', type=int)
    parser_train.add_argument(
        '--prefix', default='model', help='Prefix for saving model (default mod)')
    parser_train.set_defaults(func=train.run)

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
