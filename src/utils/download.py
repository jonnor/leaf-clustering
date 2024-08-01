
import os
import zipfile
import tempfile

def download_unpack_zip(url, out):

    if not os.path.exists(out):
        os.makedirs(out)

    with tempfile.TemporaryDirectory() as tempdir:
        archive_path = os.path.join(tempdir, 'archive.zip')
        urlretrieve(url, archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(out)

