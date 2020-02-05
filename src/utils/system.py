import errno
import importlib
import inspect
import os
import shutil
import tempfile

from contextlib import contextmanager


def get_kwarg_dict(callable):
    signature = inspect.signature(callable)
    params = dict(signature.parameters)

    arg_dict = {}
    for name, param in params.items():
        if inspect.isclass(param.default) and issubclass(param.default, inspect._empty):
            # Not a kwarg
            continue
        arg_dict[name] = param.default
    return arg_dict


def fetch_obj_from_file(dir_, module_name, obj_name):
    module_path = str(dir_).replace('/', '.') + '.' + module_name
    module = importlib.import_module(module_path)
    for attr in dir(module):
        candidate = getattr(module, attr)
        if attr == obj_name:
            return candidate


def is_strict_subclass(c1, c2, strict=True):
    if not (c1 and c2):
        return False

    if strict:
        return issubclass(c1, c2) and c1 != c2
    return issubclass(c1, c2)


def fetch_subclass_from_file(dir_, module_name, class_, strict=False, defined_in_mod=True):
    module_path = str(dir_).replace('/', '.') + '.' + module_name
    module = importlib.import_module(module_path)
    # Fetch the highest class in the inheritance heirarchy
    highest_subclass = None
    for attr in dir(module):
        candidate = getattr(module, attr)
        if not inspect.isclass(candidate):
            continue

        if is_strict_subclass(candidate, class_, strict=strict):
            if highest_subclass and is_strict_subclass(highest_subclass, candidate):
                continue
            # if defined_in_mod and candidate.__module__ != module_path:
            #     continue
            highest_subclass = candidate
    return highest_subclass


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
