
"""
Dataset loading tools for "Smartphone based Recognition of Human Activities and Postural transitions"
https://www.archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions

Uses the raw accelerometer/gyro data instead of the pre-computed features.

License: MIT
Author: Jon Nordby
"""


import os
import glob
from urllib.request import urlretrieve

import pandas
import numpy

def assert_data_correct(data):

    # check dataset size
    assert len(data) == 1122772, len(data)
    assert len(data.columns) == 3+3+1, list(data.columns)

    # check index
    assert data.index.names == ['subject', 'experiment', 'time'], data.index.names
    assert data.dtypes['activity'] == 'category'

    # check subjects
    subjects = data.index.levels[0]
    n_subjects = subjects.nunique()
    assert n_subjects == 30, (n_subjects, list(subjects.unique()))

    # check experiments
    experiments = data.index.levels[0].unique()
    assert len(experiments) == 30, (len(experiments), list(experiments))

    # check activities
    activities = data.activity.unique()
    assert len(activities) == 12+1, (len(activities), activities)

    activity_counts = data.activity.value_counts(dropna=False)
    assert activity_counts['SITTING'] == 126677, activity_counts['SITTING']

    

def load_data(path) -> pandas.DataFrame:
    """
    Load all raw IMU data and labels in the dataset

    NOTE: Takes some tens of seconds
    """

    # Prepare metadata
    activities_path = os.path.join(path, 'activity_labels.txt')
    with open(activities_path, 'r') as f:
        data = f.read()
        lines = data.split('\n')
        activities = {}
        for l in lines:
            tok = l.strip().split(' ')
            if len(tok) == 2:
                index, name = tok
                activities[int(index)] = name

    activities = pandas.Series(activities)
    assert len(activities) == 12


    # Load the data
    raw_data_path = os.path.join(path, 'RawData')
    def load_one(path, prefix):
        path = os.path.join(raw_data_path, path)
        samplerate = 50

        # Extract metadata from filename
        filename = os.path.splitext(os.path.basename(path))[0]
        tok = filename.split('_')
        assert len(tok) == 3, tok
        datatype = tok[0]
        experiment = int(tok[1].replace('exp', ''))
        assert datatype in set(('acc', 'gyro')), datatype
        subject = int(tok[2].replace('user', ''))
    
        # Load data
        columns = [f'{prefix}x', f'{prefix}y', f'{prefix}z']
        df = pandas.read_table(path, header=None, sep=' ', names=columns)
        df['subject'] = subject
        df['experiment'] = experiment
        df['time'] = numpy.arange(len(df)) * 1.0/samplerate

        # FIXME: add a time column, based on samplerate

        return df

    assert os.path.exists(raw_data_path), raw_data_path

    gyro_files = glob.glob('gyro_*.txt', root_dir=raw_data_path)
    accelerometer_files = glob.glob('acc_*.txt', root_dir=raw_data_path)

    gyro_data = pandas.concat([ load_one(p, prefix='gyro_') for p in gyro_files ])
    accelerometer_data = pandas.concat([ load_one(p, prefix='acc_') for p in accelerometer_files ])
    index_columns = ('experiment', 'time')
    data = pandas.merge(gyro_data, accelerometer_data, left_on=index_columns, right_on=index_columns)
    data['subject'] = data['subject_x']
    data = data.drop(columns=['subject_x', 'subject_y'])

    
    # Load labels
    labels_path = os.path.join(raw_data_path, 'labels.txt')
    columns = ['experiment', 'user', 'activity', 'start', 'end']
    df = pandas.read_table(labels_path, header=None, sep=' ', names=columns)
    df['activity'] = df['activity'].map(activities.to_dict())

    data['time'] = pandas.to_timedelta(data.time, unit='s')
    data = data.set_index(['subject', 'experiment', 'time']).sort_index()


    # map activities onto time series
    data['activity'] = None
    #data = data.set_index('ex')
    timestep = pandas.Timedelta(seconds=1.0/50.0)
    for _, r in df.iterrows():
        s = r.start * timestep
        e = r.end * timestep
        ss = (r.user, r.experiment, s)
        ee = (r.user, r.experiment, e)
        data.loc[ss:ee, 'activity'] = r.activity

    # Use proper types
    data['activity'] = data.activity.astype('category')

    # Sanity checks
    assert_data_correct(data)

    return data

def download_unpack_zip(url, out):

    import zipfile
    import tempfile

    if not os.path.exists(out):
        os.makedirs(out)

    with tempfile.TemporaryDirectory() as tempdir:
        archive_path = os.path.join(tempdir, 'archive.zip')
        urlretrieve(url, archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(out)


def download(out_path=None, force=False):

    exists = os.path.exists(os.path.join(out_path, 'activity_labels.txt'))
    if exists and not force:
        # already exists
        return False    

    download_url = 'https://www.archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip'

    print('Downloading dataset to', out_path)
    download_unpack_zip(download_url, out_path)
    return True

def load_packed(packed_path):

    loaded = pandas.read_parquet(packed_path)
    assert_data_correct(loaded)

    return loaded

def main():

    dataset_path = 'data/raw/uci_har_smartphone/'
    packed_path = 'data/processed/uci_har.parquet'

    downloaded = download(dataset_path)
    data = load_data(dataset_path)
    packed_dir = os.path.dirname(packed_path)
    if not os.path.exists(packed_dir):
        os.makedirs(packed_dir)
    data.to_parquet(packed_path)

    loaded = load_packed(packed_path)
    print('Raw:\t\t', dataset_path)
    print('Packed\t\t', packed_path)

if __name__ == '__main__':
    main()
