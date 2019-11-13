
import os
import tarfile

from src.utils.download import download_file
from src.utils.system import copyfile, make_temp_dir

from src.constants import MIDI_DATA_URL, MIDI_DIR

def run(args):
    with make_temp_dir('nn_seq') as dir:
        tar_dest = os.path.join(dir, 'midi_data.tar.gz')
        download_file(MIDI_DATA_URL, tar_dest, 'Downloading MIDI data...')
        print('Extracting...')
        tf = tarfile.open(tar_dest, 'r:gz')
        tf.extractall(MIDI_DIR)
        print('DONE')
