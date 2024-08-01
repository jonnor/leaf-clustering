
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

    windows = []
    for group_idx, group_df in groups:

        # make sure order is correct
        group_df = group_df.reset_index().set_index('time').sort_index()

        # create windows
        win_start = 0
        length = len(group_df)
        while win_start < length:
            win_end = win_start + window_length
            if win_end > length:
                break
            
            win = group_df.iloc[win_start:win_end].copy()
            win['window'] = win.index[0]
            assert len(win) == window_length, (len(win), window_length)

            windows.append(win)
            win_start += window_hop

    out = pandas.concat(windows)
    out = out.set_index(groupby + ['window'])

    return out

def extract_features(windows : pandas.DataFrame, columns : list[str], groupby):

    import torch

    transform = Quant()

    def extract(df, column):
        column_data = df[column].values.astype(numpy.float32)
        win = torch.from_numpy(column_data).reshape(1,1,-1)
        X = transform.fit_transform(win, Y=None)
        print('w', win.shape, win.dtype, X.shape)
        return X

    # TODO: add names to features

    features = []
    for c in columns:

        # FIXME: use vectorized torch operation instead of apply loop
        ww = windows.groupby(groupby + ['window'])
        X = ww.apply(extract, column=c)

        features.append(X)
        print(X.shape)


    out = pandas.concat(features)
    return out
    

def main():

    #dataset = 'pamap2'
    dataset = 'uci_har'

    dataset_config = {
        'uci_har': dict(groups=['subject', 'experiment']),
        'pamap2': dict(groups=['subject']),
    }

    data_dir = './data/processed'

    data_path = os.path.join(data_dir, f'{dataset}.parquet')

    data_load_start = time.time()
    data = pandas.read_parquet(data_path)


    print(data.index.names)
    print(data.columns)
    #print(data.head())

    groups = dataset_config[dataset]['groups']
    data_columns = ['acc_x', 'acc_y', 'acc_z']
    windows = extract_windows(data, window_length=128, window_hop=64, groupby=groups)

    data_load_duration = time.time() - data_load_start
    log.info('data-loaded', dataset=dataset, samples=len(windows), duration=data_load_duration)

    ww = windows.groupby(groups + ['window'])
    features = ww.apply(extract_features, columns=data_columns, groupby=groups)


if __name__ == '__main__':
    main()
