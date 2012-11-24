from locals import *

# Code from http://rosettacode.org/wiki/Power_set#Python
def list_powerset2(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])
def powerset(s):
    return frozenset(map(frozenset, list_powerset2(list(s))))

#    hyperparams = list(itertools.product(*HYPERPARAMS.values()))
#    random.shuffle(hyperparams)
#    for h in hyperparams:
#        yield dict(zip(HYPERPARAMS.keys(), h))

HYPERPARAMS = {
    "SVR": OrderedDict({
        "C": [0.1, 1, 10, 100, 1000],
        "epsilon": [0.001, 0.01, 0.1, 1.0],
        "kernel": ["rbf", "sigmoid", "linear", "poly"],
        "degree": [1,2,3,4,5],
        "gamma": [0.],
        "cache_size": [CACHESIZE],
        "shrinking": [False, True],
    }),

    "GradientBoostedRegressor": OrderedDict({
        'loss': ['ls', 'lad'],
        'learn_rate': [1., 0.1, 0.01],
        'n_estimators': [10, 100, 1000],
        'max_depth': [1, 3, 5, 10, None],
        'min_samples_split': [1, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'subsample': [0.1, 0.32, 1],
    }),

    "RandomForestRegressor": OrderedDict({
        'n_estimators': [10, 100, 1000]
        'max_depth': [1, 3, 5, 10, None],
        'min_samples_split': [1, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'min_density': [0.01, 0.1, 1.0],
        'bootstrap': [True, False],
        'oob_score': [True, False],
#        'verbose': [True],
    }),
}
