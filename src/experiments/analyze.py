
import seaborn
import pandas
import structlog
import numpy

import os.path
import uuid
import math
from pathlib import Path

from src.utils.pareto import find_pareto_front
from src.utils.latex import preview_latex

log = structlog.get_logger()


DEFAULT_STRATEGY_ORDER=[
    'original',
    'majority',
    'quantize',
    'ClusterQ8',
    'ClusterQ4',
]


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
    mm.loc[mm.leaves_per_class.notna() & (mm.leaf_bits.isna()), 'strategy'] = 'cluster-float'
    mm.loc[mm.leaves_per_class.notna() & (mm.leaf_bits == 8), 'strategy'] = 'ClusterQ8'
    mm.loc[mm.leaves_per_class.notna() & (mm.leaf_bits == 4), 'strategy'] = 'ClusterQ4'
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

def compute_perf_change(df, reference, 
        groupby='dataset',
        metric='test_roc_auc',
    ):
    df = df.copy()

    assert not df.empty
    assert not reference.empty
    ref_df = reference.reset_index().groupby(groupby).median(numeric_only=True)

    print(ref_df.head())

    # Extract change in performance
    def subtract_ref(df, metric='test_roc_auc'):
        dataset = df.name
        matches = ref_df.loc[dataset]
        #assert len(matches) == 1, matches
        ref = matches[metric]
        out = df[metric] - ref
        return out

    def divide_ref(df, metric='test_roc_auc'):
        dataset = df.name
        matches = ref_df.loc[dataset]
        #assert len(matches) == 1, matches
        ref = matches[metric]
        out = df[metric] / ref
        return out

    grouped = df.groupby(groupby, as_index=False)
    df['perf_change'] = grouped.apply(subtract_ref, include_groups=False).reset_index().set_index('id')[metric]
    df['size_change'] = grouped.apply(divide_ref, metric='total_size', include_groups=False).reset_index().set_index('id')['total_size']

    return df

def plot_overall_performance_vs_baseline(df, path=None, depth_limit='min_samples_leaf'):

    # XXX: maybe move this outside, shoud be shared for consistency?
    df = name_strategies(df)

    # what we are comparing against
    reference_experiment = 'sklearn_default_int16'
    ref = df[df.experiment.str.contains(reference_experiment)]

    # Filter down to relevant data
    experiment_prefix = 'min_samples_leaf_trees'
    data = df[df.experiment.str.contains(experiment_prefix)]
   
    # Sanity checks
    assert not ref.empty
    assert ref.trees.unique() == [100]
    assert ref[depth_limit].unique() == [1]
    # should have multiple trees
    n_trees = data['trees'].unique()
    assert len(n_trees) >= 2, n_trees
    # should have mutiple depth levels
    depth_limit_values = data[depth_limit].unique() 
    assert len(depth_limit_values) >= 4, depth_limit_values

    # Compute changes wrt baseline/reference
    data = compute_perf_change(data, ref)

    col_order = sorted(data.trees.unique(), reverse=True)
    g = seaborn.relplot(data=data.reset_index(),
        kind='line',
        x=depth_limit,
        y='perf_change',
        hue='strategy',
        col='trees',
        col_order=col_order,
        aspect=1,
        height=3,
        legend=True,
        errorbar=('ci', 95),
        estimator='median',
        #err_style='bars',
        lw=2.0,
        alpha=0.5,
    )
    g.set(xscale="log")
    #g.set(xlim=(1, 100))
    g.set(ylim=(-10, 2))
    g.refline(y=0.0, alpha=0.8, ls='--')
    #g.refline(y=-2.0, alpha=0.5, color='black')
    #g.refline(y=-4.0, alpha=0.5, color='orange')
    for i, ax in enumerate(g.axes.flatten()):
        ax.grid()
        if i >= 1:
            x_axis = ax.axes.get_xaxis()
            x_axis.get_label().set_visible(False)


    #g.figure.suptitle('Performance drop over baseline')
    g.figure.tight_layout()
    g.figure.legends = []
    g.figure.legend(loc="lower center", ncol=6)
    g.figure.savefig(path)

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)

def leaf_add_grid(g, **kwargs):
    for i, ax in enumerate(g.axes.flatten()):
        ax.grid(**kwargs)

def plot_leaf_uniqueness(df,
        path=None,
        experiment='min_samples_leaf_trees10',
        depth_limit = 'min_samples_leaf',
        strategy='original',
    ):

    # XXX: maybe move this outside, shoud be shared for consistency?
    df = name_strategies(df)

    is_data = (df.strategy == strategy) & (df.experiment.str.contains(experiment))
    data = df[is_data]

    # should be for a single tree
    n_trees = data.trees.unique()
    assert len(n_trees) == 1, n_trees

    # should have mutiple depth levels
    depth_limit_values = data[depth_limit].unique() 
    assert len(depth_limit_values) >= 4, depth_limit_values


    #fig, axs = plt.subplots(ncols=2)
    height = 3.0
    aspect = 1.5
    g = seaborn.catplot(data=data,
            x=depth_limit,
            kind='box',
            y='unique_leaves_percent', 
            height=height,
            aspect=aspect
    )

    g.set(ylim=(0, 105))
    leaf_add_grid(g, axis='y')
    g.figure.tight_layout()

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)

def plot_leaf_size_proportion(df,
        path=None,
        experiment='min_samples_leaf_trees10',
        depth_limit = 'min_samples_leaf',
        strategy='original',
    ):

    # XXX: maybe move this outside, shoud be shared for consistency?
    df = name_strategies(df)

    is_data = (df.strategy == strategy) & (df.experiment.str.contains(experiment))
    data = df[is_data]

    # should be for a single tree
    n_trees = data.trees.unique()
    assert len(n_trees) == 1, n_trees

    # should have mutiple depth levels
    depth_limit_values = data[depth_limit].unique() 
    assert len(depth_limit_values) >= 4, depth_limit_values

    height = 3.0
    aspect = 1.5
    g = seaborn.catplot(data=data,
            x=depth_limit,
            kind='box',
            y='leaf_size_percent', 
            height=height,
            aspect=aspect,
    )

    g.set(ylim=(0, 105))
    leaf_add_grid(g, axis='y')
    g.figure.tight_layout()

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)



def aggregated_performance(df,
        agg='median',
        folds = 5,
        repetitions = 1,
        metrics = ['perf_change', 'total_size'],
        check_groups_same=True,
    ):
    results_per_group = repetitions * folds

    groupby = [
        'dataset',
        'strategy',
        'min_samples_leaf',
        'leaf_bits',
        'trees',
        'leaves_per_class'
    ]

    n_folds = df.folds.unique()
    assert n_folds == [folds], n_folds

    groups = df.groupby(groupby, as_index=True, dropna=False)

    # check that we got all the results / have specified the groups correctly
    for idx, g in groups:
        if len(g) == 0:
            print('Empty group', idx)
            continue

        mismatch = len(g) != results_per_group

        leaf_bits_values = df.leaf_bits.unique()

        if mismatch and check_groups_same:
            print('Group mismatch')
            print(idx, len(g))
            print(g['id'].nunique())
            print(g['run'].unique())
            print(g.head(30))
            print(leaf_bits_values)
            raise ValueError(f'wrong numer of values per group: {len(g)}, {results_per_group}')
            
    out = groups.agg(agg, numeric_only=True)[metrics]
    return out


def performance_comparison_datasets(df,
        strategy_order=None,
        experiment=None,
        depth_limit=None,
        metric=None,
    ):

    # XXX: maybe move this outside, shoud be shared for consistency?
    df = name_strategies(df)

    data = df.copy()
    #data = data.reset_index()
    data = data[data.experiment.str.contains(experiment)]
    data = data[data.strategy.isin(strategy_order)]

    print(df.strategy.value_counts())

    # Use best-hyperparam for each dataset as the reference
    #reference_experiment = 'sklearn_default_int16'
    is_ref = (
        df.experiment.str.contains(experiment) &
        df.leaf_bits.isna() &
        df.leaves_per_class.isna()
    )
    #ref = df[is_ref]
    ref = data[data.strategy == 'original']
    ref = aggregated_performance(ref, metrics=[metric, 'total_size'])
    ref = ref.loc[ref.groupby('dataset')[metric].idxmax()]

    print(df.strategy.value_counts())

    # drop the different quantization options
    #data = data[data.leaf_bits != 8]
    #data = data[data.leaf_bits.isin([None, 8, 0])]

    # Sanity checks
    assert not ref.empty
    assert len(ref) == data.dataset.nunique(), len(ref)

    # should be for a single tree
    n_trees = data.trees.unique()
    #assert len(n_trees) == 1, n_trees

    # Compute changes wrt baseline/reference
    data = compute_perf_change(data, ref)

    # Aggregate over the folds/repretitions
    data = aggregated_performance(data)
    data = data.sort_values(['dataset', 'strategy', 'perf_change'], ascending=False)
    
    return data

def find_pareto(df,
    cost_metric='total_size',
    performance_metric='perf_change',
    min_performance=-10.0,
    ):

    pareto_params = dict(
        cost_metric=cost_metric,
        performance_metric=performance_metric,
        min_performance=min_performance,
    )
    
    pf = find_pareto_front(df, **pareto_params)
    pf['point'] = numpy.arange(len(pf))
    pf = pf.set_index('point')
    return pf

def plot_performance_datasets(df,
        path=None,
        strategy_order=None,
        experiment='min_samples_leaf_trees',
        depth_limit='min_samples_leaf',
        metric='test_roc_auc',
    ):

    if strategy_order is None:
        strategy_order = DEFAULT_STRATEGY_ORDER

    data = performance_comparison_datasets(df,
        strategy_order=strategy_order,
        experiment=experiment,
        depth_limit=depth_limit,
        metric=metric
    )

    # Plot pareto front as lines
    pareto = data.groupby(['dataset', 'strategy'], as_index=True).apply(find_pareto)
    g = seaborn.relplot(data=pareto.reset_index(),
                    kind='line',
                    col='dataset',
                    col_wrap=6,
                    y='perf_change',
                    x='total_size',
                    hue='strategy',
                    hue_order=strategy_order,
                    height=3.0,
                    aspect=1.0,
                    legend=True,
                   )

    # Add scatter plots of original data
    for ax, (_, facet_data) in zip(g.axes.flat, g.facet_data()):
        facet_dataset = facet_data.dataset.unique()
        #print(facet_dataset)
        assert len(facet_dataset) == 1
        facet_dataset = facet_dataset[0]
        scatter_data = data.loc[facet_dataset]
        seaborn.scatterplot(ax=ax,
            data=scatter_data.reset_index(),
            y='perf_change',
            x='total_size',
            hue='strategy',
            hue_order=strategy_order,
            alpha=0.5,
            legend=False,
            s=8.0,
            zorder=1,
        )
        
    # Configure scale
    g.set(xscale="log")
    g.set(ylim=(-10, 5))
    g.set(xlim=(10, 100e3))
    g.refline(y=0.0, ls='-', color='black', alpha=0.5, lw=1.0)

    # Configure legend
    g.figure.legends = []
    g.figure.legend(loc="upper center", ncol=6)
    g.figure.tight_layout()

    # Configure grid
    from matplotlib.ticker import LinearLocator, FixedLocator, MultipleLocator
    for ax in g.axes.flat:
        ax.grid(True, which='major', axis='x', linestyle='-', linewidth=1.0)
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=1.0)
        ax.grid(True, which='minor', axis='y', linestyle='-', linewidth=0.3)
        ax.yaxis.set_major_locator(MultipleLocator(5.0))
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))

    # Reduce space beween subplots, and make room for legend up top
    g.figure.subplots_adjust(wspace=0.05, hspace=0.15, top=0.98)

    if path is not None:
        g.figure.savefig(path)
        print('Wrote', path)

    return g.figure


def style_multiindex_latex(df,
        bold_metrics=None,
        threshold_metrics=None,
        decimal_places=None,
        rename_metrics=None,
        summary_rows=[],
    ):
    """
    NOTE: require these packages. xcolor,booktabs,multirow
    """

    styled = df.copy()
    decimal_places = decimal_places or {}
    rename_metrics = rename_metrics or {}

    def format_value(val, metric):
        """Format value based on metric's decimal places"""
        if pandas.isna(val):
            return ''  # Empty string instead of 'nan'
        decimals = decimal_places.get(metric, 2)
        return f"{val:.{decimals}f}"
    
    # Bold best results per row for each metric
    if bold_metrics:
        for metric, mode in bold_metrics.items():
            # Get all columns for this metric
            metric_cols = [col for col in df.columns if col[1] == metric]
            
            # For each row, find and bold the best value
            for idx in df.index:
                row_vals = df.loc[idx, metric_cols]
                best_col = row_vals.idxmax() if mode == 'max' else row_vals.idxmin()
                val = df.loc[idx, best_col]
                fmt = format_value(val, metric)
                styled.loc[idx, best_col] = f"\\textbf{{{fmt}}}"
    
    # Color by threshold
    if threshold_metrics:
        for metric, thresh in threshold_metrics.items():
            metric_cols = [col for col in df.columns if col[1] == metric]
            
            for col in metric_cols:
                for idx in styled.index:
                    val = df.loc[idx, col]
                    # Skip if already styled (bold)
                    if isinstance(styled.loc[idx, col], str):
                        continue
                    color = 'subtlegreen' if val >= thresh else 'red'
                    fmt = format_value(val, metric)
                    styled.loc[idx, col] = f"\\textcolor{{{color}}}{{{fmt}}}"
    
    # Format remaining unstyled values
    for col in styled.columns:
        metric = col[1] if isinstance(col, tuple) else col
        for idx in styled.index:
            if not isinstance(styled.loc[idx, col], str):
                val = df.loc[idx, col]
                styled.loc[idx, col] = format_value(val, metric)
    
    # Rename columns
    if rename_metrics:
        styled.columns = styled.columns.map(
            lambda x: (x[0], rename_metrics.get(x[1], x[1])) 
            if isinstance(x, tuple) 
            else rename_metrics.get(x, x)
        )
    
    # Escape underscores in column names
    styled.columns = styled.columns.map(
        lambda x: (x[0].replace('_', '\\_'), x[1].replace('_', '\\_')) 
        if isinstance(x, tuple) 
        else str(x).replace('_', '\\_')
    )
    
    # Escape underscores in index
    styled.index = styled.index.map(lambda x: str(x).replace('_', '\\_'))
    

    # Create column format with vertical lines between experiments
    experiments = df.columns.get_level_values(0).unique()
    col_groups = []
    for exp in experiments:
        # Count metrics for this experiment
        num_cols = sum(df.columns.get_level_values(0) == exp)
        col_groups.append('r' * num_cols)
    # 'l' for index, then groups separated by |
    col_format = 'l|' + '|'.join(col_groups) + '|'
    
    # Generate LaTeX
    latex_table = styled.to_latex(escape=False, multirow=True, 
                                  column_format=col_format)

    # Add horizontal line before summary rows if specified
    if summary_rows:
        lines = latex_table.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            # Check if this line contains any summary row
            if any(row.replace('_', '\\_') in line for row in summary_rows):
                # Add midrule before first summary row
                if i > 0 and '\\midrule' not in new_lines[-1]:
                    new_lines.append('\\midrule')
            new_lines.append(line)
        latex_table = '\n'.join(new_lines)


    preamble = '\\definecolor{subtlegreen}{RGB}{34, 139, 34}'
    return preamble + latex_table


def extract_best(data, lower_performance_boundary, reference='original'):

    dfs = []
    for (dataset, strategy), df in data.groupby(['dataset', 'strategy']):

        # Find the smallest models with performance above threshold
        # - else the model closest to the threshold
        filtered = df[df['perf_change'] >= lower_performance_boundary]
        result = (filtered.nsmallest(1, 'total_size', keep='all').nlargest(1, 'perf_change') 
                  if not filtered.empty 
                  else df.nlargest(1, 'perf_change')).reset_index()

        #print(dataset, strategy, df.shape)
        #print(result)
        #print(
        dfs.append(result)

    
    best = pandas.concat(dfs, ignore_index=True)

    # Compute size_saving, wrt baseline
    baseline = best[best['strategy'] == reference][['dataset', 'total_size']].rename(
        columns={'total_size': 'baseline_size'}
    )
    best = best.merge(baseline, on='dataset')
    #best['size_saving'] = ((best['baseline_size'] - best['total_size']) / best['baseline_size']) * 100

    best['size_saving'] = best['baseline_size'] / best['total_size'] 
    #best['size_saving'] = best['total_size'] / best['baseline_size']

    best = best.drop(columns=['baseline_size'])

    best['perf_change'] = best['perf_change'].round(2)
    best['total_size'] = best['total_size'].round(0)
    best['size_saving'] = best['size_saving'].round(2)

    return best

def plot_performance_datasets_table(df,
        path=None,
        strategy_order=None,
        experiment='min_samples_leaf_trees',
        depth_limit='min_samples_leaf',
        metric='test_roc_auc',
        lower_performance_boundary=-1.0,
    ):

    if strategy_order is None:
        strategy_order = DEFAULT_STRATEGY_ORDER

    # XXX: this is slows, so caching is useful for rapid iteration
    cached_path = 'foo.parquet'

    if os.path.exists(cached_path):
        data = pandas.read_parquet(cached_path)
    else:
        data = performance_comparison_datasets(df,
            strategy_order=strategy_order,
            experiment=experiment,
            depth_limit=depth_limit,
            metric=metric
        )
        data.to_parquet(cached_path)


    best = extract_best(data, lower_performance_boundary=lower_performance_boundary)
    #best = best.set_index('dataset')

    pivot = best.pivot(
        index='dataset',
        columns='strategy',
        values=['perf_change', 'size_saving', 'leaf_bits', 'leaves_per_class']
    )

    #print(pivot)

    pivot = pivot.swaplevel(axis=1).sort_index(axis=1)
    # Ensure consistent order for experiments
    pivot = pivot[strategy_order]

    # Drop not-so-useful
    pivot = pivot.drop(columns=[
        ('original', 'leaf_bits'),
        ('majority', 'leaf_bits'),
        ('ClusterQ4', 'leaf_bits'),
        ('ClusterQ8', 'leaf_bits'),
    ])

    pivot = pivot.drop(columns=[
        #('original', 'size_saving'),
        ('original', 'leaves_per_class'),
        ('majority', 'leaves_per_class'),
        ('quantize', 'leaves_per_class'),
    ])

    # Ensure consistent column order for metrics
    all_metrics = pivot.columns.get_level_values(1).unique().tolist()
    for m in ['perf_change', 'size_saving']:
        if m in all_metrics:
            all_metrics.remove(m)
    metric_order = ['perf_change', 'size_saving'] + all_metrics
    pivot = pivot.reindex(metric_order, level=1, axis=1)

    # Ensure a consistent ordering of rows
    pivot = pivot.sort_values(by=('majority', 'size_saving'), ascending=True)

    # Add summary rows
    # Calculate averages
    size_saving_cols = [col for col in pivot.columns if col[1] == 'size_saving']
    avg_row = pivot[size_saving_cols].mean().to_frame().T
    avg_row.index = ['Average']

    # Calculate wins
    best_per_row = pivot[size_saving_cols].idxmax(axis=1)
    wins_row = pandas.Series({col: (best_per_row == col).sum() for col in size_saving_cols}).to_frame().T
    wins_row.index = ['Wins']

    # Concatenate - missing columns automatically become NaN
    pivot = pandas.concat([pivot, avg_row, wins_row])

    #fig, ax = plt.subplots(
    #g = seaborn.relplot(kind='scatter', data=best, x='size_saving', y='perf_change', hue='strategy')
    #g.figure.savefig('scatter.png')

    # FIXME: use same order/colors as rest
    g = seaborn.displot(kind="ecdf",
        data=best, x='size_saving',
        hue='strategy',
        aspect=2.0,
        height=3.0,
    )   
    g.set(xlim=(0, 10.0))
    #g.refline(x=1.0, lw=2.0, ls='--', color='black')

    for i, ax in enumerate(g.axes.flatten()):
        ax.grid()
 
    g.figure.savefig('dis.png')


    latex_output = style_multiindex_latex(
        pivot,
        bold_metrics={'size_saving': 'max'},
        threshold_metrics={'perf_change': lower_performance_boundary},
        decimal_places={'perf_change': 2, 'size_saving': 2, 'total_size': 0, 'leaf_bits': 0, 'leaves_per_class': 0 },
        rename_metrics={'perf_change': 'Score', 'size_saving': 'Size', 'total_size': 'Size', 'leaf_bits': 'Bits', 'leaves_per_class': 'LC' },
        summary_rows=['Average', 'Wins'],
    )

    # Usage
    latex_output = '\\footnotesize\n' + latex_output 

    preview_path = os.path.splitext(path)[0]+'.pdf'
    packages = ['booktabs', 'multirow', 'geometry', 'multirow', 'xcolor']
    output_pdf = 'table_preview.pdf'

    if path is not None:
        open(path, 'w').write(latex_output)
        print('Wrote', path)
        preview_latex(latex_output, packages, preview_path)
        print('Wrote', preview_path)



def enrich_results(df,
        leaf_node_bytes_default=4,
        decision_node_bytes_default = 8):

    # compute storage size
    #leaf_bytes_per_class = numpy.ceil(df['leaf_bits'] / 8) # NOTE: no bit-packing, full bytes
    leaf_bytes_per_class = numpy.ceil((2*df['leaf_bits'])/8)*2 # support bit-packing 2x4 bits into 1 byte

    leaf_bytes_per_class = leaf_bytes_per_class.fillna(value=leaf_node_bytes_default).astype(float)

    decision_node_bytes = df['dtype'].map({'int16': 2, 'float': 4})
    missing_decision_node_bytes = decision_node_bytes[decision_node_bytes.isna()]
    assert len(missing_decision_node_bytes) == 0, df[decision_node_bytes.isna()][['dtype']]

    df = df.rename(columns={'test_leasize': 'test_leafsize'}) # Fixup typo

    decision_nodes = df['test_nodes'] - df['test_leaves']
    df['leaf_size'] = df['test_leafsize'] * leaf_bytes_per_class * df['test_uniqueleaves']
    df['decision_size'] = decision_nodes * decision_node_bytes
    df['total_size'] = df['leaf_size'] + df['decision_size']
    df['test_roc_auc'] = 100.0 * df['test_roc_auc'] # scale up to avoid everything being in the decimals

    df['leaf_size_percent'] = 100.0 * (df['leaf_size'] / df['total_size'])
    df['unique_leaves_percent'] = 100.0 * (df['test_uniqueleaves'] / df['test_leaves'])

    # Practical aliases
    df['trees'] = df['n_estimators']

    # Add identifier per row, for ease of merging data
    df['id'] = df.apply(lambda _: uuid.uuid4(), axis=1)
    df = df.set_index('id')

    return df

def comma_separated(s, delimiter=','):
    tok = s.split(delimiter)
    return tok

ALL_PLOTS=[
    'dataset_results_table',
    'dataset_results_pareto',
    'leaf_analysis',
    'overall_performance',
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

    if 'overall_performance' in args.plots:
        plot_overall_performance_vs_baseline(df, path='hyperparam-perfdrop-trees-strategies.png')

    if 'dataset_results_table' in args.plots:
        plot_performance_datasets_table(df, path='perf-dataset-table.tex')

    if 'dataset_results_pareto' in args.plots:
        plot_performance_datasets(df, path='perf-pareto-datasets.png')

    if 'leaf_analysis' in args.plots:
        plot_leaf_size_proportion(df, path='leaf-proportion.png')
        plot_leaf_uniqueness(df, path='leaf-uniqueness.png')

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
