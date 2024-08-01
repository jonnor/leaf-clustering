
import uuid

import pandas

def enrich_results(df,
        leaf_node_bytes_default=4,
        decision_node_bytes_default = 8):

    if 'leaf_bits' not in df.columns:
        df['leaf_bits'] = None
        df['leaf_bits'] = df.leaf_bits.astype(float) 

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

    # Add identifier per row, for ease of merging data
    df['id'] = df.apply(lambda _: uuid.uuid4(), axis=1)
    df = df.set_index('id')

    return df


df = pandas.read_parquet('har_results.parquet')

df = df.rename(columns=lambda c: c.replace('param_', ''))
df = df.rename(columns=lambda c: c.replace('mean_test_', 'test_'))
df = df.rename(columns={'n_estimators': 'trees'})

#print(df.columns)

df = enrich_results(df).reset_index()
#print(df.columns)

display_columns = [
    'trees',
    'min_samples_leaf',
    'mean_train_f1_micro',
    'test_f1_micro',
    'total_size'
]

print(df[display_columns])
