
import os
import sys

import openml
import pandas
import openml.datasets
import structlog


log = structlog.get_logger()


def download_openml_cc18(out_dir):
    """
    Download datasets that are part of the OpenML CC18 benchmark collection

    NOTE: datasets with missing values are ignored. Around 8 out of 72.

    Takes around 400 MB on disk in total.
    """

    datasets_dir = os.path.join(out_dir, 'datasets')
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    tasks_path = os.path.join(out_dir, 'tasks.csv')

    if os.path.exists(tasks_path):
        tasks = pandas.read_csv(tasks_path)

    else:
        # Get list of datasets/tasks
        suite = openml.study.get_suite(99)
        assert suite.name.startswith('OpenML-CC18')
        tasks = openml.tasks.list_tasks(output_format="dataframe")    
        tasks = tasks.query("tid in @suite.tasks")

        # Remove tasks/datasets with missing values
        tasks_missing_values = tasks['NumberOfInstancesWithMissingValues'] > 0
        tasks_too_many_features = tasks['NumberOfFeatures'] >= 255
        drop_tasks = tasks_missing_values | tasks_too_many_features
        print('Dropping tasks not fitting criteria')
        print(tasks[drop_tasks][['tid', 'did', 'name']])
        tasks = tasks[~drop_tasks]

        tasks.to_csv(tasks_path)
        log.info('task-list-downloaded', tasks=len(tasks), path=tasks_path)

    for dataset_id, dataset_name in zip(tasks['did'], tasks['name']):
        # TODO: add retrying with delay, sometimes there are connection problems

        dataset_path = os.path.join(datasets_dir, f'{dataset_id}.parquet')
        if os.path.exists(dataset_path):
            log.info('dataset-exists', id=dataset_id, name=dataset_name)
            continue

        # This is done based on the dataset ID.
        dataset = openml.datasets.get_dataset(dataset_id,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )

        target = dataset.default_target_attribute
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=target)

        data = X.copy()
        data['__target'] = y

        data.to_parquet(dataset_path)
        log.info('dataset-downloaded', id=dataset_id, name=dataset_name)


if __name__ == '__main__':
    default_path = 'data/raw/openml-cc18'
    path = default_path
    if len(sys.argv) > 1:
        path = sys.argv[1]

    download_openml_cc18(path)
