
import os
import time

import pandas
import scipy.stats
import numpy
import structlog

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from emlearn.evaluate.trees import model_size_bytes

from ..features.quant import Quant

log = structlog.get_logger()

def evaluate():

    # TODO: setup data loading
    # FIXME: setup subject-based cross validation


    # Spaces to search for hyperparameters
    hyperparameters = {
        'n_estimators': [ 5, 10 ],
        "max_features": numpy.linspace(0.30, 0.90, 10),
        'min_samples_split': [ 2**n for n in range(0, 10) ],
    }

    clf = RandomForestClassifier(random_state = 0, n_jobs=1, class_weight = "balanced")


    f1_micro = make_scorer(f1_score, average="micro")

    search = GridSearchCV(
        clf,
        param_grid=hyperparameters,
        scoring={
            # our predictive model metric
            'f1_micro': f1_micro,
            # metrics for the model costs
            'size': model_size_bytes,
        },
        refit='f1_micro',
        cv=5,
        return_train_score=True,
        n_jobs=4,
        verbose=2,
    )


    # check the results on the validation set
    #hypothesis = clf.predict(features_validation)
    #validation_score = f1_score(validation_y, hypothesis, average="micro")

    # check also the results on the test set
    hypothesis = clf.predict(features_test)
    test_score = f1_score(self.test_y, hypothesis, average="micro")

def extract_windows(sensordata : pandas.DataFrame,
    window_length : int,
    window_hop : int,
    groupby : list[str],
    ):

    groups = sensordata.groupby(groupby)


    for group_idx, group_df in groups:

        windows = []

        # make sure order is correct
        group_df = group_df.reset_index().set_index('time').sort_index()

        # create windows
        win_start = 0
        length = len(group_df)
        while win_start < length:
            win_end = win_start + window_length
            # ignore partial window at the end
            if win_end > length:
                break
            
            win = group_df.iloc[win_start:win_end].copy()
            win['window'] = win.index[0]
            assert len(win) == window_length, (len(win), window_length)

   
            windows.append(win)

            win_start += window_hop

        yield windows


def extract_features(sensordata : pandas.DataFrame,
    columns : list[str],
    groupby,
    window_length = 128,
    window_hop = 64,
    quant_div = 4,
    quant_depth = 6,
    label_column='activity',
    ):
    """
    Convert sensor data into fixed-sized time windows and extact features
    """

    import torch

    transform = Quant(div=quant_div, depth=quant_depth)

    features = []

    def assign_window_label(labels, majority=0.66):
        """
        Assign the most common label to window, if it is above the @majority threshold
        Otherwise return NA
        """

        counts = labels.value_counts()
        threshold = majority * len(labels)
        if counts.iloc[0] > threshold:
            return counts.iloc[0]
        else:
            return None

    # Split into fixed-length windows
    generator = extract_windows(sensordata, window_length, window_hop, groupby=groupby)
    for windows in generator:
    
        # Extract features
        f = []
        for c in columns:

            values = numpy.stack([w[c] for w in windows]).astype(numpy.float32)
            tensor = torch.from_numpy(values).reshape(len(values),1,window_length)
            X = transform.fit_transform(tensor, Y=None)
            f.append(X)

        ff = numpy.concatenate(f, axis=-1)
        # TODO: add names to features
        df = pandas.DataFrame(ff)

        # Attach labels
        df[label_column] = [ assign_window_label(w[label_column]) for w in windows ]

        # Combine with identifying information
        index_columns = list(groupby + ['window'])
        for idx_column in index_columns:
            df[idx_column] = [w[idx_column].iloc[0] for w in windows]
        df = df.set_index(index_columns)

        features.append(df)

    out = pandas.concat(features)
    return out
    

def main():

    dataset = 'pamap2'
    dataset = 'uci_har'

    dataset_config = {
        'uci_har': dict(
            groups=['subject', 'experiment'],
            data_columns = ['acc_x', 'acc_y', 'acc_z'],
        ),
        'pamap2': dict(
            groups=['subject'],
            data_columns = ['hand_acceleration_16g_x', 'hand_acceleration_16g_y', 'hand_acceleration_16g_z'],
        ),
    }

    data_dir = './data/processed'

    data_path = os.path.join(data_dir, f'{dataset}.parquet')

    data_load_start = time.time()
    data = pandas.read_parquet(data_path)


    #print(data.index.names)
    print(data.columns)
    #print(data.head())

    groups = dataset_config[dataset]['groups']
    data_columns = dataset_config[dataset]['data_columns']
    #windows = extract_windows(data, window_length=128, window_hop=64, groupby=groups)

    data_load_duration = time.time() - data_load_start
    log.info('data-loaded', dataset=dataset, samples=len(data), duration=data_load_duration)


    feature_extraction_start = time.time()
    features = extract_features(data, columns=data_columns, groupby=groups)

    feature_extraction_duration = time.time() - feature_extraction_start
    log.info('feature-extraction-done',
        dataset=dataset,
        instances=len(features),
        duration=feature_extraction_duration,
    )


if __name__ == '__main__':
    main()
