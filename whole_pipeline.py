import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest

njobs = 7

def ranker(train, test):
    full = np.concatenate((train, test))
    replace = {key: value for key, value in zip(sorted(np.unique(full)),
        range(len(np.unique(full))))}
    return [replace[key] for key in train], [replace[key] for key in test]

def perform_cv(estimator, train, labels):
    result = cross_val_score(estimator, train, labels, scoring='accuracy', cv=8)
    print(result)
    print(np.mean(result), np.std(result))

def voting(predictionslist):
    re = []
    for pre in predictionslist:
        re.append(round(np.mean(pre)))
    return re

def my_cool_cv(X, l, models, metric=accuracy_score, cv=5,
               shuffle=True, random_state=0, draw=True):
    """models: [('name', model instance), ...] """
    kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    scores = []
    models_scores = [[] for _ in models]

    split_i = 0
    for train_index, test_index in kf.split(X):
        # Dataset split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = l[train_index], l[test_index]

        # fit and predict
        predictions = []
        for i, (_, model) in enumerate(models):
            cur_model = model.fit(X_train, y_train)
            predictions.append(cur_model.predict(X_test))
            models_scores[i].append(metric(y_pred=predictions[i], y_true=y_test))

        # voting
        vot = voting(np.array(predictions).T)

        # Final Validation
        cur_metric = metric(y_pred=vot, y_true=y_test)
        scores.append(cur_metric)
        print('split # {}, score = {}, models scores std = {}'\
            .format(split_i, cur_metric,
            np.std([scr[split_i] for scr in models_scores])))

        split_i += 1

    print()
    print(scores)
    print(np.mean(scores), np.std(scores))
    print()

    if draw:
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(models))]
        plt.figure(figsize=(10, 7))
        for (mod_name, _), scr, color in zip(models, models_scores, colors):
            plt.plot(scr, label='{} : {}'.format(mod_name, np.mean(scr)), color=color)

        plt.plot(scores, linewidth=3, label='Voting')
        plt.legend(loc='best')

# Read data
train = pd.read_csv('x_train.csv', delimiter=';', header=None)
y = pd.read_csv('y_train.csv', header=None)
test = pd.read_csv('x_test.csv', header=None, delimiter=';')

# Filter columns
uniques = []
for col in list(train.columns):
    uniques.append(len(train[col].unique()))
filt_cols = [col for u, col in zip(uniques, list(train.columns)) if u < 600]

good_cols = filt_cols

runs_vot = []
for run_number in range(10):
    print('-------------------\nrun {}\n-------------------'.format(run_number))
    # Add DPGMM columns

    # data preparation
    good_train = train[filt_cols].copy()
    good_test = test[filt_cols].copy()

    class0 = good_train[y[0] == 0].values
    class1 = good_train[y[0] == 1].values
    class2 = good_train[y[0] == 2].values
    class3 = good_train[y[0] == 3].values
    class4 = good_train[y[0] == 4].values
    classes = [class0, class1, class2, class3, class4]

    # models initialization
    dirichlets_list = []
    for seed in np.random.randint(3, 3000, 5, dtype='int'):
        dirichlets = [
            BayesianGaussianMixture(n_components=3, n_init=3,
                init_params='random', max_iter=1000, random_state=seed),
            BayesianGaussianMixture(n_components=5, n_init=5,
                init_params='random', max_iter=1000, random_state=seed),
            BayesianGaussianMixture(n_components=5, n_init=5,
                init_params='random', max_iter=1000, random_state=seed),
            BayesianGaussianMixture(n_components=5, n_init=5, i
                nit_params='random', max_iter=1000, random_state=seed),
            BayesianGaussianMixture(n_components=3, n_init=3,
                init_params='random', max_iter=1000, random_state=seed)
                    ]
        dirichlets_list.append(dirichlets)

    # fit and predict
    col_train = [np.zeros(train.shape[0]) for _ in range(len(dirichlets_list[0]))]
    col_test = [np.zeros(test.shape[0]) for _ in range(len(dirichlets_list[0]))]
    for dirichlets in dirichlets_list:
        models = []
        for i, (klass, dpgmm) in enumerate(zip(classes, dirichlets)):
            model = dpgmm.fit(klass)
            col_train[i] += model.score_samples(good_train[filt_cols].values)
            col_test[i] += model.score_samples(good_test[filt_cols].values)

    scores_cols = []
    for i, (tr_col, te_col) in enumerate(zip(col_train, col_test)):
        name = 'score{}'.format(i)
        good_train[name] = np.array(tr_col) / len(dirichlets_list)
        good_test[name] = np.array(te_col) / len(dirichlets_list)
        scores_cols.append(name)

    # rank
    for col in good_cols:
        ranked = ranker(good_train[col].values, good_test[col].values)
        good_train[col], good_test[col] = ranked

    # add poly columns
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=False)
    poly_train = pd.DataFrame(poly.fit_transform(good_train[filt_cols].values))
    poly_test = pd.DataFrame(poly.fit_transform(good_test[filt_cols].values))
    poly_train = pd.concat([poly_train, good_train[scores_cols]], axis=1)
    poly_test = pd.concat([poly_test, good_test[scores_cols]], axis=1)

    # Isolation Forest
    isfo = IsolationForest(max_features=1.0, n_jobs=njobs,
                        random_state=np.random.randint(1, 100, 1, dtype='int')[0],
                         contamination=0.05, bootstrap=True)
    mo = isfo.fit(train[filt_cols].values)
    poly_train['isofor'] = mo.decision_function(train[filt_cols].values)
    poly_test['isofor'] = mo.decision_function(test[filt_cols].values)

    mo = isfo.fit(good_train[filt_cols].values)
    poly_train['isofor1'] = mo.decision_function(good_train[filt_cols].values)
    poly_test['isofor1'] = mo.decision_function(good_test[filt_cols].values)

    # add noise columns
    for i in np.random.randint(2, 2000, 1, dtype='int'):
        np.random.seed(i)
        poly_train['noise{}'.format(i)] = np.random.normal(0, 1, poly_train.shape[0])
        poly_test['noise{}'.format(i)] = np.random.normal(0, 1, poly_test.shape[0])

    # init models and perform cv
    models = []
    for i, seed in enumerate(np.random.randint(2, 2000, 2)):
        models.append(('ext{}'.format(i),
            ExtraTreesClassifier(n_estimators=900, criterion='entropy', max_depth=16,
                                   bootstrap=False, n_jobs=njobs, random_state=seed)))
        models.append(('rf{}'.format(i),
            RandomForestClassifier(n_estimators=800, criterion='entropy', max_depth=14,
                                   n_jobs=njobs, random_state=seed, class_weight='balanced')))
        models.append(('xgb{}'.format(i),
                       XGBClassifier(max_depth=10, learning_rate=0.1,
                                        objective='multi:softmax', n_estimators=40,
                                         subsample=0.7, gamma=0.01, seed=seed)))
        models.append(('lgbm{}'.format(i),
                      LGBMClassifier(num_leaves=90, n_estimators=50,
                        objective='multiclass', subsample=0.7,
                                    max_depth=20, learning_rate=0.1)))

    my_cool_cv(poly_train.values, y[0].values.flatten(),
        models=models, random_state=10, draw=False)

    # fit and predict
    X_train = poly_train.values
    y_train = y[0].values.flatten()
    X_test = poly_test.values

    predictions = []
    for i, (_, model) in enumerate(models):
        cur_model = model.fit(X_train, y_train)
        predictions.append(cur_model.predict(X_test))

    # voting
    vot = voting(np.array(predictions).T)

    # write answer
    with open('stochastic{}.csv'.format(run_number), 'w') as buf:
        for pre in vot:
            buf.write(str(pre) + '\n')

    runs_vot.append(vot)

# voting
vot = voting(np.array(runs_vot).T)

# write answer
with open('stochastic_voting1.csv', 'w') as buf:
    for pre in vot:
        buf.write(str(pre) + '\n')
