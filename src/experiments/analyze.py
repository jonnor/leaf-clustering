
import seaborn
import pandas
import structlog

import uuid
import math
from pathlib import Path

log = structlog.get_logger()

def plot_leaf_quantization(df, path):

    # TODO: also plot KDE and/or histogram
    # Isolate experiments that only change leaf quantization
    df = df[df.leaves_per_class.isna()]

    # Extract change in performance wrt no change
    def rel_perf(df, metric='test_roc_auc'):
        matches = df[df.leaves_per_class.isna() & df.leaf_bits.isna() & (df.min_samples_leaf == 1)]
        assert len(matches) == 1, matches
        ref = matches.iloc[0][metric]
        out = df[metric] - ref
        return out

    rel = df.groupby(
        ['dataset', 'split'], as_index=False
    ).apply(rel_perf, include_groups=False).reset_index().set_index('id')['test_roc_auc']
    df['perf_change'] = rel

    assert 'perf_change' in df.columns
    df = df.dropna(subset=['leaf_bits'])
    df['leaf_bits'] = df['leaf_bits'].astype('int').replace({0: '0 (majority vote)'})

    # Plot results
    g = seaborn.catplot(data=df, kind='strip',
        x='leaf_bits', y='perf_change',
        height=5, aspect=2.0,
    )
    g.refline(y=0.1)
    g.refline(y=-0.1)
    #g.set(xlim=(0.50, 1.0))

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)

    return g


def plot_leaf_clustering(df, path):

    # isolate experiments that are only changing clustering
    df = df[df.leaf_bits.isna()]

    
    # Extract change in performance
    def rel_perf(df, metric='test_roc_auc'):
        matches = df[df.leaves_per_class.isna() & df.leaf_bits.isna() & (df.min_samples_leaf == 1)]
        assert len(matches) == 1, matches
        ref = matches.iloc[0][metric]
        out = df[metric] - ref
        return out

    rel = df.groupby(
        ['dataset', 'split'], as_index=False
    ).apply(rel_perf, include_groups=False).reset_index().set_index('id')['test_roc_auc']
    df['perf_change'] = rel

    assert 'perf_change' in df.columns
    df = df.dropna(subset=['leaves_per_class'])

    # Plot results
    g = seaborn.catplot(data=df, kind='strip',
        x='leaves_per_class', y='perf_change',
        height=5, aspect=2.0,
    )
    g.refline(y=0.1)
    g.refline(y=-0.1)
    #g.set(xlim=(0.50, 1.0))

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)

    return g

def name_strategies(df):

    mm = df.copy()
    mm['strategy'] = 'other'
    mm.loc[mm.leaves_per_class.isna() & mm.leaf_bits.isna(), 'strategy'] = 'original'
    mm.loc[mm.leaves_per_class.isna() & (mm.leaf_bits.notna()), 'strategy'] = 'quantize'
    mm.loc[mm.leaves_per_class.notna() & (mm.leaf_bits.isna()), 'strategy'] = 'cluster'
    mm.loc[mm.leaves_per_class.notna() & (mm.leaf_bits.notna()), 'strategy'] = 'joint'
    mm.loc[mm.leaves_per_class.isna() & (mm.leaf_bits == 0.0), 'strategy'] = 'majority'

    return mm

def plot_perf_vs_size(df, path):

    # Drop data
    df = df[df.leaf_bits != 4]    

    # Extract change in performance
    def subtract_ref(df, metric='test_roc_auc'):
        matches = df[df.leaves_per_class.isna() & df.leaf_bits.isna() & (df.min_samples_leaf == 1)]
        assert len(matches) == 1, matches
        ref = matches.iloc[0][metric]
        out = df[metric] - ref
        return out

    def divide_ref(df, metric='test_roc_auc'):
        matches = df[df.leaves_per_class.isna() & df.leaf_bits.isna() & (df.min_samples_leaf == 1)]
        assert len(matches) == 1, matches
        ref = matches.iloc[0][metric]
        out = df[metric] / ref
        return out

    grouped = df.groupby(['dataset', 'split'], as_index=False)
    df['perf_change'] = grouped.apply(subtract_ref, include_groups=False).reset_index().set_index('id')['test_roc_auc']
    df['size_change'] = grouped.apply(divide_ref, metric='total_size', include_groups=False).reset_index().set_index('id')['total_size']

    mm = df.groupby(['leaf_bits', 'leaves_per_class'], dropna=False).median(numeric_only = True).reset_index()
    mm = name_strategies(mm)

    #assert 'perf_change' in df.columns
    #df = df.dropna(subset=['leaves_per_class'])

    # Plot results
    g = seaborn.relplot(data=mm, kind='scatter',
        x='size_change', y='perf_change', hue='strategy',
        height=5, aspect=2.0,
    )
    g.refline(y=0.0)
    #g.set(xlim=(0.50, 1.0))

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)

    return g



def plot_size_improvement(df, path, optimizer_param='leaves_per_class'):

    # Filter data
    pass

    # Extract change in performance
    def subtract_ref(df, metric='test_roc_auc'):
        matches = df[df.leaves_per_class.isna() & df.leaf_bits.isna() & (df.min_samples_leaf == 1)]
        assert len(matches) == 1, matches
        ref = matches.iloc[0][metric]
        out = df[metric] - ref
        return out

    def divide_ref(df, metric='test_roc_auc'):
        matches = df[df.leaves_per_class.isna() & df.leaf_bits.isna() & (df.min_samples_leaf == 1)]
        assert len(matches) == 1, matches
        ref = matches.iloc[0][metric]
        out = df[metric] / ref
        return out

    grouped = df.groupby(['dataset', 'split'], as_index=False)
    df['perf_change'] = grouped.apply(subtract_ref, include_groups=False).reset_index().set_index('id')['test_roc_auc']
    df['size_change'] = grouped.apply(divide_ref, metric='total_size', include_groups=False).reset_index().set_index('id')['total_size']

    df = name_strategies(df)


    #df = df[df.strategy == 'joint']
    df = df[df.leaf_bits == 8]
    #df = df[df.perf_change >= -1.0]

    #df.groupby(['dataset', ''])

    # make categorical
    df[optimizer_param] = df[optimizer_param].astype('Int64').astype(str)

    best = df.groupby(['dataset', optimizer_param], dropna=False).median(numeric_only=True).reset_index()

    def find_best(df):
        s = df.sort_values('size_change', ascending=True)
        b = s.iloc[0]
        return b


    best = best.groupby(['dataset']).apply(find_best)

    # Plot results
    g = seaborn.relplot(data=best, kind='scatter',
        x='size_change',
        y='perf_change',
        hue=optimizer_param,
        height=6, aspect=2.0, #s=5.0,
    )
    g.refline(y=0.0)
    g.set(xlim=(0, 1.0))

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)

    return g



def enrich_results(df,
        leaf_node_bytes_default=4,
        decision_node_bytes_default = 8):

    # compute storage size
    leaf_bytes_per_class = df['leaf_bits'] / 8
    leaf_bytes_per_class = leaf_bytes_per_class.fillna(value=leaf_node_bytes_default).astype(int)

    # FIXME: take the feature precision into account
    decision_node_bytes = decision_node_bytes_default

    df = df.rename(columns={'test_leasize': 'test_leafsize'}) # Fixup typo

    decision_nodes = df['test_nodes'] - df['test_leaves']
    df['leaf_size'] = df['test_leafsize'] * leaf_bytes_per_class * df['test_uniqueleaves']
    df['decision_size'] = decision_nodes * decision_node_bytes
    df['total_size'] = df['leaf_size'] + df['decision_size']

    df['test_roc_auc'] = 100.0 * df['test_roc_auc'] # scale up to avoid everything being in the decimals

    # Add identifier per row, for ease of merging data
    df['id'] = df.apply(lambda _: uuid.uuid4(), axis=1)
    df = df.set_index('id')

    return df

def comma_separated(s, delimiter=','):
    tok = s.split(delimiter)
    return tok

ALL_PLOTS=[
    'size_improvement',
    'perf_vs_size',
    'leaf_quantization',
    'leaf_clustering',
]

def parse():
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--results', metavar='DIRECTORY', type=str, default='./output/results/experiments.parquet',
                        help='Where the input data is stored')

    # FIXME: change default to reports/figures
    parser.add_argument('--out-dir', metavar='DIRECTORY', type=str, default='./',
                        help='Where to store figures')

    parser.add_argument('--plots', type=comma_separated, default=ALL_PLOTS,
                        help='Which plots to make')

    args = parser.parse_args()

    return args


def main():
    args = parse()

    # Check inputs
    invalid_plots = set(args.plots) - set(ALL_PLOTS)
    if invalid_plots:
        raise ValueError(f'Unknown plots {invalid_plots}')

    # Load data
    df = pandas.read_parquet(args.results)

    print(df.shape)
    print(list(sorted(df.columns)))
    print(df.head(1))

    # Enrich
    df = enrich_results(df)


    print(df.experiment.value_counts())
    print(df.run.value_counts())
    print(df.leaves_per_class.unique())
    print(df.leaf_bits.unique())


    out_dir = Path(args.out_dir)

    # FIXME: have a better way of marking the baseline to compare with
    if 'size_improvement' in args.plots:
        plot_size_improvement(df,
            path=out_dir/'size-improvement.png',
            optimizer_param='leaf_bits',
        )

    if 'perf_vs_size' in args.plots:
        plot_perf_vs_size(df, path=out_dir/'size-change.png')

    if 'leaf_quantization' in args.plots:
        plot_leaf_quantization(df, path=out_dir /'leaf-quantization.png')

    if 'leaf_clustering' in args.plots:
        plot_leaf_clustering(df, path=out_dir/'leaf-clustering.png')


    log.debug('analyze-plots-done', out=out_dir)



if __name__ == '__main__':
    main()
