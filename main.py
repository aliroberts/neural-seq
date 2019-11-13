
import argparse

from src.commands import encode, decode, fetch_data, generate, train

def main():
    parser = argparse.ArgumentParser(prog='main')
    subparsers = parser.add_subparsers()

    # Decoding of string format to MIDI file
    parser_decode = subparsers.add_parser('decode')
    parser_decode.set_defaults(func=decode.run)

    # Encoding of MIDI files to string format (for training)
    parser_encode = subparsers.add_parser('encode')
    parser_encode.set_defaults(func=encode.run)

    # Data fetching
    parser_fetch = subparsers.add_parser('fetch-data')
    parser_fetch.set_defaults(func=fetch_data.run)

    # Training
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--data-dir', required=True)
    parser_train.add_argument('--dropout', default=1, help='Dropout multiplier', type=float)    
    parser_train.add_argument('--pre-trained', help='Model with which to start training')
    parser_train.add_argument('--bs', default=32, help='Batch size', type=int)
    parser_train.add_argument('--bptt', default=200, help='BPTT', type=int) 
    parser_train.add_argument('--em_sz', default=300, help='Embedding size', type=int) 
    parser_train.add_argument('--nh', default=600, help='Number of hidden activations', type=int) 
    parser_train.add_argument('--nl', default=4, help='Number of LSTM layers', type=int) 
    parser_train.add_argument('--min_freq', default=1, help='Minimum frequency of token for it to be included', type=int) 
    parser_train.add_argument('--epochs', default=10, help='Epochs to train model for', type=int),
    parser_train.add_argument('--save_freq', default=1, help='Frequency at which to save model snapshots (every SAVE_FREQ epochs)', type=int)
    parser_train.add_argument('--prefix', default='model', help='Prefix for saving model (default mod)') 
    parser_train.set_defaults(func=train.run)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()