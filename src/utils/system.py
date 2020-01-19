import errno
import importlib
import inspect
import os
import shutil
import tempfile

from contextlib import contextmanager


def get_kwarg_dict(callable):
    argspec = inspect.getfullargspec(callable)
    defaults = argspec.defaults
    all_args = argspec.args

    arg_dict = {arg: default for arg, default in zip(
        all_args[len(all_args) - len(defaults):], defaults)}
    return arg_dict


def fetch_obj_from_file(dir_, module_name, obj_name):
    module_path = str(dir_).replace('/', '.') + '.' + module_name
    module = importlib.import_module(module_path)
    for attr in dir(module):
        candidate = getattr(module, attr)
        if attr == obj_name:
            return candidate


def fetch_class_from_file(dir_, module_name, class_, strict=False):
    module_path = str(dir_).replace('/', '.') + '.' + module_name
    module = importlib.import_module(module_path)
    for attr in dir(module):
        candidate = getattr(module, attr)
        try:
            subclass = issubclass(candidate, class_) if not strict else issubclass(
                candidate, class_) and candidate != class_

            if subclass:
                return candidate
        except TypeError:
            continue


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
