
import os
import glob
import uuid
import copy

import sklearn.base
from sklearn.metrics import get_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.special

from ..utils.parallel import ProgressParallel, joblib
from . import metrics
from ..utils.config import config_number_list

from emlearn.preprocessing.quantizer import Quantizer
import emlearn
print('emlearn', emlearn.__path__)


import pandas
import numpy
import structlog

log = structlog.get_logger()



def quantize_probabilities(p, bits=8, range_check=True):
    assert bits <= 32
    assert bits >= 1
    max = numpy.max(p)
    min = numpy.min(p)
    if range_check:
        assert max <= 1.0, max
        assert min >= 0.0, min

    steps = (2**bits)

    quantized = (p * steps).round(0).astype(numpy.uint32)
    out = quantized.astype(float) / steps

    return out

def get_leaves(estimator):

    assert type(estimator) == RandomForestClassifier, type(estimator)

    ll = []
    for e in estimator.estimators_:
        is_leaf = (e.tree_.children_left == -1) & (e.tree_.children_right == -1)
        l = e.tree_.value[is_leaf]
        ll.append(l)

    leaves = numpy.concatenate(ll)
    return leaves

def reshape_leaves_as_features(leaves):

    # reshape to 2d
    vv = leaves
    if len(vv.shape) == 3:
        assert vv.shape[1] == 1, 'only single output supported'
        vv = vv[:, 0, :]
    #vv = vv.reshape(-1, 1) if len(vv.shape) == 1 else vv
    assert len(vv.shape) == 2, (vv.shape, leaves.shape)

    return vv




def optimize(estimator, n_samples, n_classes, leaf_quantization=None, leaves_per_class=None):
    """
    NOTE: mutates @estimator
    """

    # Find leaves
    leaves = get_leaves(estimator)
    assert leaves.shape[1] == 1, 'only single output supported'
    leaves = numpy.squeeze(leaves)

    # Replace predict with function that handles quantized values,
    # and still gives valid probabilities as output
    # Otherwise get ValueError: Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes
    original_predict_proba = estimator.predict_proba
    def predict_proba(est, X):
        orig = original_predict_proba(X)
        proba = scipy.special.softmax(orig, axis=1)
        return proba
    estimator.predict_proba = lambda X: predict_proba(estimator, X)

    # Quantize leaves
    n_unique_leaves_quantized = None
    if leaf_quantization is not None:

        for e in estimator.estimators_:

            is_leaf = (e.tree_.children_left == -1) & (e.tree_.children_right == -1)
            values = e.tree_.value
            assert values.shape[1] == 1
        
            if leaf_quantization == 0:
                # simple voting. highest probability gets 1.0, rest 0.0
                # in practice only returning the index of the most probable class
                voted_class = numpy.argmax(values, axis=-1)

                quantized = numpy.zeros_like(values)
                for i, c in enumerate(voted_class):
                    quantized[i, 0, c] = 1.0

            else:
                quantized = quantize_probabilities(values, bits=leaf_quantization)
            assert quantized.shape == values.shape

            for i in range(len(e.tree_.value)):
                is_leaf = (e.tree_.children_left[i] == -1) and (e.tree_.children_right[i] == -1)
                if is_leaf:
                    e.tree_.value[i] = quantized[i]

        leaves = get_leaves(estimator)
        assert leaves.shape[1] == 1, 'only single output supported'
        leaves = numpy.squeeze(leaves)
        n_unique_leaves_quantized = len(numpy.unique(leaves, axis=0))
        
        #log.debug('leaf quantized',
        #    bits=self.leaf_quantization,
        #    unique_after=n_unique_leaves_quantized,
        #    unique_before=n_unique_leaves,
        #)
        

    # Cluster leaves
    n_leaves = len(leaves)
    n_unique_leaves = len(numpy.unique(leaves, axis=0))
    if leaves_per_class is None:
        max_leaves = None
    else:
        max_leaves = int(leaves_per_class * n_classes)
        max_leaves = min(max_leaves, n_unique_leaves)
        if n_unique_leaves_quantized is not None:
            max_leaves = min(max_leaves, n_unique_leaves_quantized)
        max_leaves = min(max_leaves, n_samples)

    if max_leaves is None:
        return None

    #print('clusters', max_leaves, n_unique_leaves, n_classes, n_samples, max_leaves)

    if (n_unique_leaves <= n_classes) or (n_unique_leaves <= max_leaves):
        # assume already optimial
        pass

    else:

        cluster = KMeans(n_clusters=max_leaves, tol=1e-4, max_iter=100)
        cluster.fit(reshape_leaves_as_features(leaves))

        # Replace by closest centroid
        for e in estimator.estimators_:

            is_leaf = (e.tree_.children_left == -1) & (e.tree_.children_right == -1)
            values = e.tree_.value
            vv = reshape_leaves_as_features(values)
            #print('v', vv.shape, cluster.cluster_centers_.shape)
            #print(cluster.cluster_centers_)
            c_idx = cluster.predict(vv)
            centroids = cluster.cluster_centers_[c_idx]
            # ensure that stored leaves are quantized
            # cluster centers are continious and might be in-between quantized input values
            if leaf_quantization is not None:
                centroids = quantize_probabilities(centroids, bits=leaf_quantization, range_check=False)
            #print('SS', centroids.shape)
            #print('cc', len(numpy.unique(centroids, axis=0)), len(numpy.unique(c_idx)))
            # XXX: is this correct ??
            v = numpy.where(numpy.expand_dims(is_leaf, -1), centroids, numpy.squeeze(values))
            v = numpy.reshape(v, values.shape)
            for i in range(len(e.tree_.value)):
                e.tree_.value[i] = v[i]

    # check post-conditions
    out_leaves = numpy.squeeze(get_leaves(estimator))
    out_leaves_unique = numpy.unique(out_leaves, axis=0)
    n_unique = len(out_leaves_unique)
    if False:
        log.debug('optimize-check',
            samples=n_samples,
            unique=n_unique,
            max_leaves=max_leaves,
            quantization=leaf_quantization, leaves_per_class=leaves_per_class,
        )
    if leaf_quantization is not None:
        if leaf_quantization == 0:
            # should be majority. One leaf per class
            assert n_unique <= n_classes, (n_unique, n_classes)
        else:
            # should already be quantized to N bits
            out_leaves_q = quantize_probabilities(out_leaves, bits=leaf_quantization, range_check=False)
            assert (out_leaves == out_leaves_q).all()

    if leaves_per_class is not None:
        # should not have more than max_leaves
        assert n_unique <= max_leaves, (n_unique, max_leaves, out_leaves_unique)

    # Estimator has been modified
    return None

def setup_data_pipeline(data, quantizer=None):

    target_column = '__target'
    feature_columns = list(set(data.columns) - set([target_column]))
    Y = data[target_column]
    X = data[feature_columns]

    # data preprocessing
    cat_columns = make_column_selector(dtype_include=[object, 'category'])(X)
    num_columns = list(set(feature_columns) - set(cat_columns))

    # ensure that all categories have a well defined mapping, regardless of train/test splits
    categories = OrdinalEncoder().fit(X[cat_columns]).categories_

    log.debug('setup-data-pipeline',
        samples=len(X),
        quantizer=quantizer,
        categorical=len(cat_columns),
        numerical=len(num_columns),
    )

    if quantizer:
        num_transformer = make_pipeline(RobustScaler(), quantizer)
    else:
        num_transformer = make_pipeline(RobustScaler())

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_columns),
        ('cat', OrdinalEncoder(categories=categories), cat_columns)
    ])

    return X, Y, preprocessor


def flatten(l):
     flat = []
     for items in l:
         flat.extend(items)
     return flat

def cross_validate(pipeline, X, Y,
        cv=10,
        n_jobs=4,
        repetitions=1,
        verbose=1,
        optimizers=[{'quantize': None, 'cluster': None}],
    ):

    scoring = {
        'nodes': metrics.tree_nodes,
        'leaves': metrics.tree_leaves,
        'leasize': metrics.leaf_size,
        'uniqueleaves': metrics.unique_leaves,
        'roc_auc': get_scorer('roc_auc_ovo_weighted'),
    }

    n_classes = len(Y.unique())

    def run_one(split, train_index, test_index):
        X_train = X.iloc[train_index]
        Y_train = Y.iloc[train_index]
        X_test = X.iloc[test_index]
        Y_test = Y.iloc[test_index]

        estimator = sklearn.base.clone(pipeline)
        estimator.fit(X_train, Y_train)

        dfs = []

        for options in optimizers:
            res = {}
            opt = copy.deepcopy(estimator)

            q = options['quantize']
            c = options['cluster']
            res['leaves_per_class'] = c
            res['leaf_bits'] = q
            res['split'] = split
    
            # run model optimizations
            classifier = opt.named_steps['randomforestclassifier']
            optimize(classifier, n_samples=len(Y_train), n_classes=n_classes, leaf_quantization=q, leaves_per_class=c)

            # evaluation
            for metric, scorer in scoring.items():
                res[f'test_{metric}'] = scorer(opt, X_test, Y_test)

            for metric in ['roc_auc']:
                scored = scoring[metric]
                res[f'train_{metric}'] = scorer(opt, X_train, Y_train)


            dfs.append(res)

        return dfs 
            

    splitter = RepeatedStratifiedKFold(n_splits=cv, n_repeats=repetitions, random_state=1)
    jobs = [ joblib.delayed(run_one)(i, train_index, test_index) for i, (train_index, test_index) in enumerate(splitter.split(X, Y)) ]

    executor = ProgressParallel(n_jobs=n_jobs, verbose=verbose, total=len(jobs))
    out = executor(jobs)

    df = pandas.DataFrame.from_records(flatten(out))        

    return df

def run_dataset(pipeline, dataset_path, quantizer=None,
    n_jobs = 4,
    repetitions = 1,
    optimizers={},
    cv=10,
    scoring = 'roc_auc_ovo_weighted',
    ):

    data = pandas.read_parquet(dataset_path)
    X, Y, preprocessor = setup_data_pipeline(data, quantizer=quantizer)

    # combine into a pipeline
    pipeline = make_pipeline(
        preprocessor,
        *pipeline,
    )

    df = cross_validate(pipeline, X, Y, cv=cv, n_jobs=n_jobs, repetitions=repetitions, optimizers=optimizers)

    
    return df

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def run_datasets(pipeline, out_dir, run_id, quantizer=None, kvs={}, dataset_dir=None, **kwargs):

    if dataset_dir is None:
        dataset_dir = 'data/raw/openml-cc18/datasets'

    matches = glob.glob('*.parquet', root_dir=dataset_dir)
    assert len(matches) == 52, len(matches)
    for no, f in enumerate(matches):
        dataset_id = os.path.splitext(f)[0]
        dataset_path = os.path.join(dataset_dir, f)

        res = run_dataset(pipeline, dataset_path, quantizer=quantizer, **kwargs)
        res['dataset'] = str(dataset_id)
        for k, v in kvs.items():
            res[k] = v
        res['run'] = run_id

        #o = os.path.join(out_dir, f'dataset={dataset_id}')
        #ensure_dir(o)
        o = os.path.join(out_dir, f'r_{run_id}_ds_{dataset_id}.part')
        assert not os.path.exists(o), o
        res.to_parquet(o)

        score = res.groupby(['leaves_per_class', 'leaf_bits'], dropna=False)['test_roc_auc'].median().sort_values(ascending=False)

        log.info('dataset-run-end', dataset=dataset_id, dataset_no=no, **kvs)
        print(score)


def get_depth_limiter():

    depth_limiters = dict(
        min_samples_leaf = config_number_list('MIN_SAMPLES_LEAF', ''),
        min_samples_split = config_number_list('MIN_SAMPLES_SPLIT', ''),
        max_depth = config_number_list('MAX_DEPTH', ''),
        max_leaf_nodes = config_number_list('MAX_LEAF_NODES', ''),
        min_impurity_decrease = config_number_list('MIN_IMPURITY_DECREASE', ''),
    )

    depth_limiter = None # tuple: (metric, value)
    depth_limiters_configured = [ k for k, v in depth_limiters.items() if v ]
    assert len(depth_limiters_configured) == 1, depth_limiters_configured
    metric = depth_limiters_configured[0]
    value = depth_limiters[metric]

    return metric, value

def main():
    
    repetitions = int(os.environ.get('REPETITIONS', '3'))
    folds = int(os.environ.get('FOLDS', '5'))
    trees = config_number_list('TREES', '100')
    experiment = os.environ.get('EXPERIMENT', 'tree-minsamplesleaf')
    feature_dtype = os.environ.get('FEATURE_DTYPE', None)

    depth_limiter_metric, depth_limiter_values = get_depth_limiter()

    experiments = {}
    for t in trees:

        for limit in depth_limiter_values:
            name = f'{experiment}-{t}-{limit}'
            config = dict(n_estimators=t, dtype=feature_dtype)
            config[depth_limiter_metric] = limit
            if feature_dtype == 'int16':
                config['target_max'] = (2**15)-1
            experiments[name] = config

    print('Experiments:')
    for k, v in experiments.items():
        print(k, v)

    # Leaf clustering, with or without quantization
    cluster_quantizers = [None, 0, 4, 8]
    clusters = [ 1, 2, 4, 8, 16, 32 ] # XXX: None is skipped, since it comes after
    optimizers = [ {'quantize': q, 'cluster': c} for q in cluster_quantizers for c in clusters ]

    # Leaf quantization only, no clustering
    quantizers = [ None, 0, 1, 2, 3, 4, 5, 6, 7, 8 ]
    optimizers += [ {'quantize': q, 'cluster': None} for q in quantizers ]

    log.info('experiments-ready',
        experiments=len(experiments),
        optimizers=len(optimizers),
    )

    out_dir = 'output/results/experiments.parquet'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for experiment, config in experiments.items():

        log.info('experiment-start', experiment=experiment, **config)

        p = []

        # feature quantization (optional)
        quantizer = None
        if config.get('dtype'):
            quantizer = Quantizer(dtype=config['dtype'],
                max_quantile=0.001, out_max=config['target_max'])

        rf_hyperparams_names = set(config) - set(['target_max', 'dtype'])
        rf_hyperparams = { k: config[k] for k in rf_hyperparams_names }

        # classifier
        rf = RandomForestClassifier(**rf_hyperparams)
        p.append(rf)

        run_id = uuid.uuid4().hex.upper()[0:6] + f'_{experiment}'

        kvs = {}
        kvs.update(config)
        kvs['experiment'] = experiment
        kvs['folds'] = folds
        kvs['repetitions'] = repetitions

        run_datasets(p, quantizer=quantizer, optimizers=optimizers, kvs=kvs,
            out_dir=out_dir, run_id=run_id, repetitions=repetitions, cv=folds)


    print('Results written to', out_dir)

if __name__ == '__main__':
    main()
