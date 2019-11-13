
from pathlib import Path

DATA_DIR = Path('.data')

# Store the extracted MIDI files here
MIDI_DIR = DATA_DIR/'midi_data'

# The default directory for extracting our encoded dataset to
DATASET_DIR = DATA_DIR/'dataset'
TRAINING_SET_DIR = DATASET_DIR/'train'
VALIDATION_SET_DIR = DATASET_DIR/'valid'
TEST_SET_DIR = DATASET_DIR/'test'

# MIDI files (cleaned and sorted into artist directories)
MIDI_DATA_URL = 'http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz'