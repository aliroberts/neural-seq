import errno
import os
import shutil
import tempfile

from contextlib import contextmanager


def fetch_class_from_file(dir, module_name, class_):
    module_path = str('src.models.rnn').replace('/', '.') + module_name
    module = importlib.import_module(module_path)
    for attr in dir(module):
        candidate = getattr(module, attr)
        if isinstance(candidate, class_):
            return candidate


def copyfile(src, dest):
    # Preserve symlinks
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dest)
    else:
        shutil.copy(src, dest)


def dir_names(dir):
    dirs = []
    for fname in os.listdir(dir):
        if os.path.isdir(os.path.join(os.path.abspath(dir), fname)):
            dirs.append(fname)
    return dirs


def ensure_dir_exists(path):
    """ Ensure that a directory at the given path exists """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@contextmanager
def make_temp_dir(suffix=''):
    """ Use this within a `with` statement. Cleans up after itself. """
    path = tempfile.mkdtemp(suffix=suffix)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def yn(msg):
    try:
        valid_response = False
        while not valid_response:
            response = input(msg) or 'y'
            if response == 'y' or response == 'Y':
                return True
            elif response == 'n' or response == 'N':
                return False
            else:
                msg = 'Please enter \'y\' or \'n\': '
    except KeyboardInterrupt:
        print()
        return False
