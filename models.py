#imports
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
import numpy as np

from modAL.disagreement import vote_entropy_sampling
from modAL.models import ActiveLearner, Committee

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from tqdm import tqdm
from collections import namedtuple

from joblib import Parallel, delayed
import itertools as it


#Data
df = pd.read_csv("data/card_transdata.csv")

#drop empty columns
df = df.dropna(axis=1, how='all')

#log transform
df['distance_from_home'] = np.log(df['distance_from_home'])
df['distance_from_last_transaction'] = np.log(df['distance_from_last_transaction'])
df['ratio_to_median_purchase_price'] = np.log(df['ratio_to_median_purchase_price'])

#Data transform
df['distance_from_home'] = (df['distance_from_home'] - np.mean(df['distance_from_home']))/np.std(df['distance_from_home'])
df['distance_from_last_transaction'] = (df['distance_from_last_transaction'] - np.mean(df['distance_from_last_transaction']))/np.std(df['distance_from_last_transaction'])
df['ratio_to_median_purchase_price'] = (df['ratio_to_median_purchase_price'] - np.mean(df['ratio_to_median_purchase_price']))/np.std(df['ratio_to_median_purchase_price'])

print(*df.columns.values)

#split to legit and fraud frames
legit, fraud = [x for _, x in df.groupby(df['fraud'] == 1)]

#pc transform data
#PC transform data
n_comp = 7
pca = PCA(n_components=n_comp)
pca.fit(df.loc[:, df.columns != 'fraud'])

legit, fraud = pca.transform(legit.loc[:, legit.columns != 'fraud']), pca.transform(fraud.loc[:, fraud.columns != 'fraud'])

#initiate data into train and test
y = df['fraud'].to_numpy()
X = df.drop('fraud', axis='columns').to_numpy()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,shuffle=True)
SEED = 58 # Set our RNG seed for reproducibility.

#count 1 in y_train
print("1 in y_train: ", np.count_nonzero(y_train == 0))

print(len(y_train))
n_queries = 100 # You can lower this to decrease run time

# You can increase this to get error bars on your evaluation.
# You probably need to use the parallel code to make this reasonable to compute
n_repeats = 5

permutations=[np.random.permutation(X_train.shape[0]) for _ in range(n_repeats)]

for i in range(n_repeats):
    if np.count_nonzero(y_train[permutations[0][0:2]] == 1) == 0:
        index = np.where(y_train == 1)[0][0]
        permutations[i][0] = index
    if np.count_nonzero(y_train[permutations[0][0:2]] == 0) == 0:
        index = np.where(y_train == 0)[0][0]
        permutations[i][0] = index

print(y_train[permutations[0][:2]])

ResultsRecord = namedtuple('ResultsRecord', ['estimator', 'query_id', 'score'])

#Committee
ModelClass=LogisticRegression

def train_committee(i_repeat, i_members, X_train, y_train, n_start):
    committee_results = []
    print('') # progress bars won't be displayed if not included

    X_pool = X_train.copy()
    y_pool = y_train.copy()

    start_indices = permutations[i_repeat][:n_start]

    committee_members = [ActiveLearner(estimator=ModelClass(),
                                       X_training=X_train[start_indices, :],
                                       y_training=y_train[start_indices],
                                       ) for _ in range(i_members)]

    committee = Committee(learner_list=committee_members,
                          query_strategy=vote_entropy_sampling)

    X_pool = np.delete(X_pool, start_indices, axis=0)
    y_pool = np.delete(y_pool, start_indices)

    for i_query in tqdm(range(1, n_queries), desc=f'Round {i_repeat} with {i_members} members', leave=False):
        query_idx, query_instance = committee.query(X_pool)

        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        committee._set_classes() #this is needed to update for unknown class labels

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

        # score = committee.score(X_test, y_test)
        pred = committee.predict(X_test)
        score = sklearn.metrics.adjusted_rand_score(y_test, pred)

        committee_results.append(ResultsRecord(
            f'committe_{i_members}',
            i_query,
            score))

    return committee_results

#random
random_results = []

ModelClass=LogisticRegression

n_start = 5


for i_repeat in range(n_repeats):
    start_points = permutations[i_repeat][:n_start]
    selected_points = start_points
    random_learner = ModelClass()
    for i_query in tqdm(range(1,n_queries)):
        query_indices=permutations[i_repeat][:n_start+i_query]
        #random leaner
        random_learner = random_learner.fit(X=X_train[query_indices, :], y=y_train[query_indices])
        # random_score = random_learner.score(X_test, y_test)
        random_score = sklearn.metrics.adjusted_rand_score(y_test, random_learner.predict(X_test))
        
        random_results.append(ResultsRecord('random', i_query, random_score))
        #committee leaner
    #committee_results = train_committee(i_repeat, 12, X_train, y_train,n_start)

n_members=[2, 4, 8]
result = Parallel(n_jobs=-1)(delayed(train_committee)(i,i_members,X_train,y_train, n_start) for i, i_members in it.product(range(n_repeats), n_members))

#uncertain

uncertain_results = []
ModelClass=LogisticRegression

n_start = 2
addn = 1

for i_repeat in range(n_repeats):
    uncertain_learner = ModelClass()
    us_query_indices=permutations[i_repeat][:n_start]
    for i_query in tqdm(range(1,n_queries)):
        
        #random leaner
        uncertain_learner = uncertain_learner.fit(X=X_train[us_query_indices, :], y=y_train[us_query_indices])

        pred = uncertain_learner.predict_proba(X_test)
        uncertain_score = sklearn.metrics.adjusted_rand_score(y_test, np.argmax(pred, axis=1))
        uncertain_results.append(ResultsRecord('uncertain', i_query, uncertain_score))
        
        pred = uncertain_learner.predict_proba(X_train)
        for _ in range(addn):
            uncertain_index = np.argmax(1-np.max(pred, axis=1))
            pred = np.delete(pred, uncertain_index, axis=0)
            us_query_indices = np.append(query_indices, uncertain_index)


print('All jobs done')
committee_results=[r for rs in result for r in rs]

df_results = pd.concat([pd.DataFrame(results)
                        for results in
                        [random_results, committee_results, uncertain_results]])

df_results_mean=df_results.groupby(['estimator','query_id']).mean()
df_results_std=df_results.groupby(['estimator','query_id']).std()

df_mean=df_results_mean.reset_index().pivot(index='query_id', columns='estimator', values='score')
df_std=df_results_std.reset_index().pivot(index='query_id', columns='estimator', values='score')

df_mean.plot(figsize=(8.5,6), yerr=df_std)
plt.grid('on')
plt.show()