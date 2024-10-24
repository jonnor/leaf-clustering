
import numpy
import pandas

def get_tree_estimators(estimator):
    """
    Get the DecisionTree instances from ensembles or single-tree models
    """

    if hasattr(estimator, 'named_steps'):
        # is a Pipeline, try to find the classifier
        estimator = estimator.named_steps['randomforestclassifier']

    if hasattr(estimator, 'estimators_'):
        trees = [ e for e in estimator.estimators_]
    else:
        trees = [ estimator ]
    return trees

def tree_nodes(model, a=None, b=None):
    """
    Number of nodes total
    """
    trees = get_tree_estimators(model)
    nodes = [ len(e.tree_.children_left) for e in trees ]
    return numpy.sum(nodes)

def tree_leaves(model, a=None, b=None):
    """
    Total number of leaf nodes
    """
    trees = get_tree_estimators(model)
    leaves = [ numpy.count_nonzero((e.tree_.children_left == -1) & (e.tree_.children_right == -1)) for e in trees ]
    return numpy.sum(leaves)

def feature_counts(model, a=None, b=None):
    """
    How many times different features are used
    """
    trees = get_tree_estimators(model)
    features = []
    for e in trees:
        features += list(e.tree_.feature)
    features = pandas.Series(features)    
    features = features[features >= 0]

    counts = features.value_counts()
    return counts

def unique_features(model, a=None, b=None):
    """
    Total number of features utilized
    """
    counts = feature_counts(model)
    return len(counts)

def unique_leaves(model, a=None, b=None):
    """
    Number of unique leaf nodes
    """
    trees = get_tree_estimators(model)

    ll = []
    for e in trees:
        l = e.tree_.value[(e.tree_.children_left == -1) & (e.tree_.children_right == -1)]
        ll.append(l)
    leaves = numpy.squeeze(numpy.concatenate(ll))

    return len(numpy.unique(leaves, axis=0))

def leaf_size(model, a=None, b=None):
    """
    Average size of leaves
    """
    trees = get_tree_estimators(model)
    sizes = [ e.tree_.value[(e.tree_.children_left == -1) & (e.tree_.children_right == -1)].shape[-1] for e in trees ]
    return numpy.median(sizes)
