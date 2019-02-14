import argparse
import collections

import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def main():
    options = argparse.ArgumentParser(description='Predict whether the documents should be expanded for a given '
                                                  'query. Note that the features and oracle must align on both query '
                                                  'and index.')
    options.add_argument('features', help='Must be <query, index, feature1, ...> tuples.')
    options.add_argument('oracle', help='Must be <query, index, value> triples.')
    args = options.parse_args()

    metrics = ['jaccard', 'expansion_rm', 'target_rm', 'doc_vs_expansion', 'pairwise', 'doc_clarity', 'exp_doc_clarity']
    features = pandas.read_csv(args.features, sep=' ', header=None, names=['query', 'index'] + metrics)
    oracle = pandas.read_csv(args.oracle, sep=',', header=None, names=['query', 'index', 'map'])

    k_fold_splitter = StratifiedKFold(n_splits=10)
    estimator = LogisticRegression(solver='lbfgs')

    confidences = collections.defaultdict(dict)
    for collection in features['index'].unique():
        # Make a copy because we're doing this once for each collection
        current_features = features.copy()

        # Unify all other collections, this is the "rest" in 1-vs-rest. Rename them to "other" and use the average
        # value across all of them.
        current_features.loc[current_features['index'] != collection, 'index'] = 'other'
        current_features = current_features.groupby(['query', 'index'], as_index=False).agg({metric: 'mean' for
                                                                                             metric in metrics})

        # The goal is to convert each feature into a single value, which is the difference between the target
        # collection value and the value for "other." To enable this, first melt the data frame so that the metric
        # name is a row variable, rather than a column.
        current_features = current_features.melt(id_vars=['query', 'index'], var_name='metric')

        # Next, pivot the table back with the two indexes (target and other) as the only two columns, one row per
        # query/metric pair. We have to use pivot_table to have a multi-index, and pivot_table requires an aggfunc,
        # but there is really only one value, so max is chosen for speed.
        current_features = pandas.pivot_table(current_features, values='value', columns='index', index=['query',
                                                                                                        'metric'],
                                              aggfunc=max)

        # Now that corresponding feature values for each index are side-by-side, it's simple to subtract one from the
        # other.
        current_features = current_features \
            .assign(diff=current_features[collection] - current_features['other']) \
            .drop([collection, 'other'], axis='columns')

        # Finally, get us back to where we started, with each metric in its own column, and each row corresponding
        # only to the query.
        current_features = pandas \
            .pivot_table(current_features, columns='metric', index='query', values='diff', aggfunc=max) \
            .reset_index()

        # Merge with the oracle so our classifier will have ready access to the correct label, and rename the target
        # collection to A and all others as B. This allows for consistent ordering with predict_proba below.
        current_features = pandas.merge(current_features, oracle).drop('map', axis='columns')
        current_features.loc[current_features['index'] != collection, 'index'] = 'B'
        current_features.loc[current_features['index'] == collection, 'index'] = 'A'

        for train_indexes, test_indexes in k_fold_splitter.split(current_features[metrics], current_features['index']):
            features_train, features_test = current_features.loc[train_indexes, metrics], current_features.loc[
                test_indexes, metrics]
            labels_train, labels_test = current_features.loc[train_indexes, 'index'], current_features.loc[
                test_indexes, 'index']

            estimator.fit(features_train, labels_train)
            probs = estimator.decision_function(features_test)
            for i, prob in enumerate(probs):
                test_index = test_indexes[i]
                query = current_features.loc[test_index, 'query']
                confidences[query][collection] = prob

    best = max if estimator.classes_[1] == 'A' else min
    predictions = [best(query_confs[1].keys(), key=lambda col: query_confs[1][col]) for query_confs in sorted(
        confidences.items(), key=lambda x: x[0])]  # ensure queries are sorted and then pick highest collection

    print(predictions)


if __name__ == '__main__':
    main()
