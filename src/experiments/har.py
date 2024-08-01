
import os
import time
import uuid

import pandas
import scipy.stats
import numpy
import structlog

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit

from emlearn.evaluate.trees import model_size_bytes

from ..features.quant import Quant
from ..experiments import metrics

log = structlog.get_logger()

def evaluate(windows : pandas.DataFrame, groupby, random_state=1, n_splits=5, label_column='activity'):

    # Setup subject-based cross validation
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=random_state)

    # Spaces to search for hyperparameters
    hyperparameters = {
        'n_estimators': [ 10 ],
        "max_features": [ 0.30 ],
        'min_samples_leaf': [ 2**n for n in range(0, 10) ],
    }

    clf = RandomForestClassifier(random_state = random_state, n_jobs=1, class_weight = "balanced")

    f1_micro = 'f1_micro'

    search = GridSearchCV(
        clf,
        param_grid=hyperparameters,
        scoring={
            # our predictive model metric
            'f1_micro': f1_micro,

            # metrics for the model costs
            'nodes': metrics.tree_nodes,
            'leaves': metrics.tree_leaves,
            'leasize': metrics.leaf_size,
            'uniqueleaves': metrics.unique_leaves,
        },
        refit='f1_micro',
        cv=splitter,
        return_train_score=True,
        n_jobs=4,
        verbose=2,
    )

    feature_columns = sorted(set(windows.columns) - set([label_column]))
    assert 'subject' not in feature_columns
    assert 'activity' not in feature_columns
    X = windows[feature_columns]
    Y = windows[label_column]
    groups = windows.index.get_level_values(groupby)
    search.fit(X, Y, groups=groups)

    results = pandas.DataFrame(search.cv_results_)

    return results


def extract_windows(sensordata : pandas.DataFrame,
    window_length : int,
    window_hop : int,
    groupby : list[str],
    ):

    groups = sensordata.groupby(groupby, observed=True)


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

def assign_window_label(labels, majority=0.66):
    """
    Assign the most common label to window, if it is above the @majority threshold
    Otherwise return NA
    """
    counts = labels.value_counts(dropna=True)
    threshold = majority * len(labels)
    if counts.iloc[0] > threshold:
        return counts.index[0]
    else:
        return None

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
    
def run_pipeline(run, dataset, data_dir, out_dir):

    dataset_config = {
        'uci_har': dict(
            groups=['subject', 'experiment'],
            data_columns = ['acc_x', 'acc_y', 'acc_z'],
            classes = [
                #'STAND_TO_LIE',
                #'SIT_TO_LIE',
                #'LIE_TO_SIT',
                #'STAND_TO_SIT',
                #'LIE_TO_STAND',
                #'SIT_TO_STAND',
                'STANDING', 'LAYING', 'SITTING',
                'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
            ],
        ),
        'pamap2': dict(
            groups=['subject'],
            data_columns = ['hand_acceleration_16g_x', 'hand_acceleration_16g_y', 'hand_acceleration_16g_z'],
            classes = [
                #'transient',
                'walking', 'ironing', 'lying', 'standing',
                'Nordic_walking', 'sitting', 'vacuum_cleaning',
                'cycling', 'ascending_stairs', 'descending_stairs',
                'running', 'rope_jumping',
            ],
        ),
    }

    if not os.path.exits(out_dir):
        os.makedirs(out_dir)

    data_path = os.path.join(data_dir, f'{dataset}.parquet')

    data_load_start = time.time()
    data = pandas.read_parquet(data_path)

    #print(data.index.names)
    #print(data.columns)

    groups = dataset_config[dataset]['groups']
    data_columns = dataset_config[dataset]['data_columns']
    enabled_classes = dataset_config[dataset]['classes']

    data_load_duration = time.time() - data_load_start
    log.info('data-loaded', dataset=dataset, samples=len(data), duration=data_load_duration)

    feature_extraction_start = time.time()
    features = extract_features(data, columns=data_columns, groupby=groups)
    labeled = numpy.count_nonzero(features['activity'].notna())

    feature_extraction_duration = time.time() - feature_extraction_start
    log.info('feature-extraction-done',
        dataset=dataset,
        total_instances=len(features),
        labeled_instances=labeled,
        duration=feature_extraction_duration,
    )

    # Drop windows without labels
    features = features[features.activity.notna()]

    # Keep only windows with enabled classes
    features = features[features.activity.isin(enabled_classes)]

    print('Class distribution\n', features['activity'].value_counts(dropna=False))

    # Run train-evaluate
    results = evaluate(features, groupby='subject')

    # Save results
    results['dataset'] = dataset
    results['run'] = run
    results_path = os.path.join(out_dir, f'r_{run}_{dataset}.results.parquet')
    results.to_parquet(results_path)
    print('Results written to', results_path)

    return results

def parse():
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--out', metavar='FILE', type=str, default='',
                        help='Where to put results')

    parser.add_argument('--dataset', type=str, default='uci_har',
                        help='Which dataset to use')
    parser.add_argument('--data-dir', metavar='DIRECTORY', type=str, default='./data/processed',
                        help='Where the input data is stored')
    parser.add_argument('--out-dir', metavar='DIRECTORY', type=str, default='./output/results/har',
                        help='Where to store results')

    args = parser.parse_args()

    return args


def main():

    args = parse()
    dataset = args.dataset
    out_dir = args.out_dir
    data_dir = args.data_dir

    run_id = uuid.uuid4().hex.upper()[0:6]

    results = run_pipeline(dataset=args.dataset,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        run=run_id,
    )

    df = results.rename(columns=lambda c: c.replace('param_', ''))
    display_columns = [
        'n_estimators',
        'min_samples_leaf',
        'mean_train_f1_micro',
        'mean_test_f1_micro',
    ]

    print('Results\n', df[display_columns])

if __name__ == '__main__':
    main()
