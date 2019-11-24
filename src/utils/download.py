import os
import sys
import requests
import shutil
import tempfile

from src import NeuralSeqException
from src.utils.system import copyfile, ensure_dir_exists


def format_size(size):
    # Takes bytes and returns a string formatted for humans
    units = ['', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if abs(size) < 1024:
            if unit == '' or unit == 'KB':
                return '%i%s' % (size, unit)
            else:
                return '%3.1f%s' % (size, unit)
        size /= 1024.0
    return '> 1024 TB'


def get_download_size(url):
    try:
        # We don't want to download the package just yet
        response = requests.head(url, timeout=10)
    except requests.exceptions.RequestException as e:
        print('GET: %s' % url)
        raise NeuralSeqException(str(e))
    size = response.headers.get('content-length')
    if not size:
        size = response.headers.get('Content-Length')
    if not response.ok or not size or not size.isdigit():
        raise NeuralSeqException('Bad response from server. '
                                 'Please try again later.')
    return int(size)


def download_file(url, destination, download_msg=None):
    if not destination[0] == '/':
        destination = os.path.join(os.getcwd(), destination)

    response = requests.get(url, stream=True, timeout=(10, None))
    content_length = response.headers.get('content-length')

    if not response.ok:
        err = 'Could not download file %s (server responded with status ' \
              'code %s)' % (url, response.status_code)
        response.close()
        raise NeuralSeqException(err)

    tmp = tempfile.mkdtemp()
    tmp_dest = os.path.join(tmp, os.path.basename(destination))
    try:
        with open(tmp_dest, 'w+b') as f:
            if download_msg:
                print(download_msg)

            if not content_length:
                f.write(response.content)
                return
            else:
                content_length = int(content_length)

            progress = 0
            bar_width = 50  # Length in chars

            for data in response.iter_content(chunk_size=1024):
                progress += len(data)
                f.write(data)
                percentage = round((progress / content_length) * 100, 1)
                bar = int(bar_width * (progress / content_length))
                stats = '%s%% (%s/%s)' % (percentage, format_size(progress),
                                          format_size(content_length))
                # Include spaces at the end so that if the stat string shortens
                # previously printed text isn't visible
                sys.stdout.write('\r[%s%s] %s ' % ('=' * bar,
                                                   ' ' * (bar_width - bar),
                                                   stats))
                sys.stdout.flush()
        response.close()
        dest_dir = os.path.dirname(destination)
        ensure_dir_exists(dest_dir)
        copyfile(tmp_dest, destination)
        print('\nDownload complete')
    except KeyboardInterrupt:
        print('\nDownload interrupted')
        raise
    finally:
        shutil.rmtree(tmp)
        response.close()
