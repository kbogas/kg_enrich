import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, cosine_similarity, chi2_kernel
from scipy.sparse.csgraph import laplacian
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from collections import Counter
from grakel.datasets import fetch_dataset
from grakel.utils import cross_validate_Kfold_SVM
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import time


def get_prime_adjacency(dict_):
    """
    Helper function to create adjacency matrices with primes, using grakel dicts as input.
    Currently we only utilize the edges. The dict has key, values in the form of:
    (head, tail): relation
    :param dict_: dict, The dict has key, values in the form of (head, tail): relation .
    From grakel second argument per graph.
    :return adj: np.array, with the prime-transformed relation numbers
    """
    nodes = set()
    rels = set()
    for key, value in dict_.items():
        nodes.add(key[0])
        nodes.add(key[1])
        rels.add(value)

    nodes = sorted(nodes)
    id2node = {}
    node2id = {}
    for i, n in enumerate(nodes):
        id2node[i] = n
        node2id[n] = i

    rels = sorted(rels)
    from sympy import nextprime
    relid2prime = {}
    prime2relid = {}
    current_int = 2
    for rel in rels:
        cur_prime = nextprime(current_int)
        relid2prime[rel] = cur_prime
        prime2relid[cur_prime] = rel
        current_int = cur_prime

    adj = np.zeros((len(nodes), len(nodes)))
    for key, value in dict_.items():
        adj[node2id[key[0]], node2id[key[1]]] = relid2prime[value]
    return adj

class ProductPower(TransformerMixin):

    def __init__(self, power=1, use_laplace=True, normalize=False, grakel_compatible=True, kernel='rbf'):
        self.power = power
        self.use_laplace = use_laplace
        self.normalize = normalize
        self.grakel_compatible = grakel_compatible
        self.kernel_str = kernel
        if self.kernel_str == 'rbf':
            self.kernel_fn = rbf_kernel
        elif self.kernel_str == 'linear':
            self.kernel_fn = linear_kernel
        elif self.kernel_str == 'cosine':
            self.kernel_fn = cosine_similarity
        elif self.kernel_str == 'chi2':
            self.kernel_fn = chi2_kernel
        else:
            raise AttributeError(f'Kernel {self.kernel_str} is not understood!')
        self.x_train = None
        self.max_ = 1


    def fit(self, X, y=None):
        x_tr = []
        for x in X:
            if self.use_laplace:
                x = laplacian(x)
            try:
                max_array = np.max(x)
            except ValueError:
                # this is for empty adjacency like TOX21_AR
                x_tr.append(np.array([0 for _ in range(self.power)]))
                continue
            if max_array > self.max_:
                self.max_ = max_array
            if self.grakel_compatible:
                n = x.shape[0]
                cur_feat = []
                cur_x = x.copy()
                # Power = 0 (1-hop, original Adjacency)
                cur_prod = np.sum(np.log(cur_x[cur_x >0]))

                if self.normalize:
                    cur_prod = cur_prod / ((n**2)*np.log(self.max_))
                cur_feat.append(cur_prod)
                for p in range(1, self.power):
                    cur_x = np.matmul(cur_x, x)
                    cur_prod = np.sum(np.log(cur_x[cur_x >0]))
                    if self.normalize:
                        cur_prod = cur_prod / ((n**2)*np.log(self.max_**p))
                    cur_prod -= cur_feat[-1]
                    cur_feat.append(cur_prod)
                x_tr.append(np.array(cur_feat))
        if self.grakel_compatible:
            x_tr = np.array(x_tr)
            self.x_train = x_tr#self.kernel_fn(x_tr, x_tr)#x_tr
        #print(self.x_train.shape)
        return self

    def transform(self, X):
        x_tr = []
        for x in X:
            if self.use_laplace:
                x = laplacian(x)
            n = x.shape[0]
            if n == 0:
                x_tr.append(np.array([0 for _ in range(self.power)]))
                continue
            cur_feat = []
            cur_x = x.copy()
            # Power = 0 (1-hop, original Adjacency)
            cur_prod = np.sum(np.log(cur_x[cur_x >0]))
            if self.normalize:
                cur_prod = cur_prod / ((n**2)*np.log(self.max_))
            cur_feat.append(cur_prod)
            for p in range(1, self.power):
                cur_x = np.matmul(cur_x, x)
                cur_prod = np.sum(np.log(cur_x[cur_x >0]))
                if self.normalize:
                    cur_prod = cur_prod / ((n**2)*np.log(self.max_**p))
                cur_prod -= cur_feat[-1]
                cur_feat.append(cur_prod)
            x_tr.append(np.array(cur_feat))
        x_tr = np.array(x_tr)
        if self.grakel_compatible:
            #print('TRANFORM: ', x_tr.shape, self.x_train.shape)
            if self.kernel_str == 'rbf':
                gamma = 1 / (x_tr.shape[1] * self.x_train.var())
                x_tr = self.kernel_fn(x_tr, self.x_train, gamma=gamma)
            else:
                x_tr = self.kernel_fn(x_tr, self.x_train)
            #X_diag = np.einsum('ij,ij->i', self.x_train, self.x_train)
            #Y_diag = np.einsum('ij,ij->i', x_tr_orig, x_tr_orig)
            #print(X_diag.shape, Y_diag.shape)
            #print(np.outer(Y_diag, X_diag).shape)
            #print(x_tr.shape)
            #x_tr /= np.sqrt(np.outer(Y_diag, X_diag))
        return x_tr




class Decomposer(TransformerMixin):

    def __init__(self, power=1, n_components=10, use_laplace=True, grakel_compatible=True, kernel='rbf'):
        self.power = power
        self.n_components = n_components
        self.use_laplace = use_laplace
        self.decomposition_method = IncrementalPCA
        self.grakel_compatible = grakel_compatible
        self.kernel_str = kernel
        if self.kernel_str == 'rbf':
            self.kernel_fn = rbf_kernel
        elif self.kernel_str == 'linear':
            self.kernel_fn = linear_kernel
        elif self.kernel_str == 'cosine':
            self.kernel_fn = cosine_similarity
        elif self.kernel_str == 'chi2':
            self.kernel_fn = chi2_kernel
        else:
            raise AttributeError(f'Kernel {self.kernel_str} is not understood!')
        self.x_train = None
        self.eps = 0.000001



        #self.decompose = PCA(n_components='mle')

    def fit(self, X, y=None):
        x_tr = []
        if self.n_components == 'average':
            self.n_components = int(np.mean([x.shape[0] for x in X if x.shape[0] > 0]))
        if self.grakel_compatible:
            for x in X:
                if self.use_laplace:
                    x = laplacian(x, normed=True, use_out_degree=True)
                n = x.shape[0]
                if n == 0:
                    x_tr.append(np.array([[self.eps for _ in range(self.n_components)] for _ in range(self.power)]))
                    continue
                cur_feat = []
                cur_x = x.copy()
                # Power = 0 (1-hop, original Adjacency)
                cur_components = self.n_components if self.n_components < cur_x.shape[1] else cur_x.shape[1] - 1
                zeros_to_append = [self.eps for _ in range(self.n_components - cur_components)]
                td = self.decomposition_method(cur_components)
                _ = td.fit_transform(cur_x)
                cur_prod = td.singular_values_[:cur_components].tolist()
                cur_feat.append(np.array(cur_prod + zeros_to_append))
                for p in range(1, self.power):
                    cur_x = np.matmul(cur_x, x)
                    td = self.decomposition_method(cur_components)
                    _ = td.fit_transform(cur_x)
                    cur_prod = td.singular_values_[:cur_components].tolist()
                    cur_feat.append(np.array(cur_prod + zeros_to_append))
                x_tr.append(np.array(cur_feat).flatten())
            #print(f'In train')
            x_tr = np.array(x_tr)
            self.x_train = x_tr#self.kernel_fn(x_tr, x_tr)#x_tr
        #print(self.x_train.shape)
        return self

    def transform(self, X):
        x_tr = []
        for x in X:
            #print(f'x.shape:{x.shape}')
            if self.use_laplace:
                x = laplacian(x, normed=True, use_out_degree=True)
            n = x.shape[0]
            if n == 0:
                x_tr.append(np.array([[self.eps for _ in range(self.n_components)] for _ in range(self.power)]).flatten())
                continue
            cur_feat = []
            cur_x = x.copy()
            # Power = 0 (1-hop, original Adjacency)
            cur_components = self.n_components if self.n_components < n else n
            zeros_to_append = [self.eps for _ in range(self.n_components - cur_components)]
            td = self.decomposition_method(cur_components)
            _ = td.fit_transform(cur_x)
            cur_prod = td.singular_values_[:cur_components].tolist()
            cur_feat.append(np.array(cur_prod + zeros_to_append))
            for p in range(1, self.power):
                cur_x = np.matmul(cur_x, x)
                td = self.decomposition_method(cur_components)
                _ = td.fit_transform(cur_x)
                cur_prod = td.singular_values_[:cur_components].tolist()
                cur_feat.append(np.array(cur_prod + zeros_to_append))
            # for cur_ in cur_feat:
            #     print(cur_.shape)
            x_tr.append(np.array(cur_feat).flatten())
        #print(Counter([item.shape[0] for item in x_tr]))
        x_tr = np.array(x_tr)
        if self.grakel_compatible:
            #print('TRANFORM: ', x_tr.shape, self.x_train.shape)
            #print(x_tr.shape, n, self.x_train.shape, self.x_train.shape)
            if self.kernel_str == 'rbf':
                gamma = 1 / (x_tr.shape[1] * self.x_train.var())
                x_tr = self.kernel_fn(x_tr, self.x_train, gamma=gamma)
            else:
                x_tr = self.kernel_fn(x_tr, self.x_train)
            #X_diag = np.einsum('ij,ij->i', self.x_train, self.x_train)
            #Y_diag = np.einsum('ij,ij->i', x_tr_orig, x_tr_orig)
            #print(X_diag.shape, Y_diag.shape)
            #print(np.outer(Y_diag, X_diag).shape)
            #print(x_tr.shape)
            #x_tr /= np.sqrt(np.outer(Y_diag, X_diag))
        return x_tr


from grakel.datasets.base import dataset_metadata

# We want edge labels as these are the multi-relational
# We need node labels for the other methodologies
# Tox datasets seem to have problems
wanted_dataset_names = []
for dataset_name, values in dataset_metadata.items():
    if values['el'] and values['nl'] and not('Tox' in dataset_name):
        wanted_dataset_names.append(dataset_name)

# Tox21_AHR has an empty graph,
# Tox21_AR is wronly packaged (the downloaded file after unzipping is named something else and throws error
# Cuneiform soemthing wrong with the node labels being strings instead of ints?
# Zinc and Alchemy do not have graph labels. Maybe need to download train,test,valid
unwanted = set(['Cuneiform', 'Tox21_AHR', 'Tox21_AR', 'ZINC_full', 'alchemy_full'])
extra_wanted = []#['MOLT-4', 'YEAST']
for dataset_name in extra_wanted:
    dataset_metadata[dataset_name] = {'el':True, 'nl':True}

wanted_dataset_names =[dt for dt in wanted_dataset_names if not(dt in unwanted)]
wanted_dataset_names = extra_wanted + wanted_dataset_names

#wanted_dataset_names = extra_wanted
print(f'Total: {len(wanted_dataset_names)} datasets')
for dt in wanted_dataset_names:
    print(dt)


import time
from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, GraphletSampling, EdgeHistogram
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

models = [
    #VertexHistogram(normalize=True),
    #ShortestPath(normalize=True),
    #WeisfeilerLehman(n_iter=1, base_graph_kernel=VertexHistogram, normalize=True),
    # WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True),
    #WeisfeilerLehmanOptimalAssignment(n_jobs=-1, n_iter=3),
    #GraphletSampling(n_jobs=-1, k=3),
    ProductPower(power=1, normalize=False, use_laplace=False, grakel_compatible=True),
    ProductPower(power=1, normalize=False, use_laplace=True, grakel_compatible=True),
    ProductPower(power=1, normalize=False, use_laplace=True, grakel_compatible=True, kernel='linear'),
    ProductPower(power=1, normalize=False, use_laplace=False, grakel_compatible=True),
    ProductPower(power=3, normalize=False, grakel_compatible=True),
    ProductPower(power=3, normalize=False, use_laplace=False, grakel_compatible=True),
    #ProductPower(power=3, normalize=False, grakel_compatible=True, kernel='linear'),
    #Decomposer(power=1, n_components='average', grakel_compatible=True),
    #Decomposer(power=3, n_components='average', grakel_compatible=True),
    #Decomposer(power=5, n_components='average', grakel_compatible=False),
    #ProductPower(power=5, grakel_compatible=False),
    #ProductPower(power=10, grakel_compatible=False),
]

model_names = [
    #'VH',
    #'SP',
    #'WL-1',
    #'WL-4',
    #'WL-OA',
    #'GR-3',
    'PP-1NN-NL',
    'PP-1NN',
    'PP-1NN-linear',
    'PP-1NN-NL-linear',
    'PP-3NN',
    'PP-3NN-NL',
    'PP-3NN-linear'

    #'DD-1_avg',
    #'DD-3_avg',
    #'DD-5_avg',
    # 'PP-5',
    # 'PP-10',
]

results = []
for dataset_name in wanted_dataset_names[:]:
    print(f'Dataset: {dataset_name}')
    if dataset_name in extra_wanted:
        dataset = read_wrapper(name=dataset_name)
    else:
        dataset = fetch_dataset(dataset_name, verbose=False, download_if_missing=True)
    G, y = dataset.data, dataset.target
    print(len(G))
    print(f'Parsed {dataset_name}')
    # Splits the dataset into a training and a test set
    # G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42, stratify=y)
    prime_adj_all = [get_prime_adjacency(g[2]) for g in  G]
    #prime_adj_train = [get_prime_adjacency(g[2]) for g in  G_train]
    #prime_adj_test = [get_prime_adjacency(g[2]) for g in G_test]
    for model, model_name in zip(models, model_names):
        #print(model_name)
        time_s = time.time()
        if 'PP' in model_name or 'DD' in model_name:
            X = prime_adj_all
            X = model.fit_transform(X)
            clf = SVC(kernel="precomputed", class_weight='balanced')
            #K_train = model.fit_transform(X_train)
            #K_test = model.transform(X_test)
            pipe = Pipeline([
                ('clf', clf)
            ])
        else:
            if dataset_name in extra_wanted:
                clf = SVC(kernel="precomputed", class_weight='balanced')
            else:
                clf = SVC(kernel="precomputed", class_weight='balanced')
            #X_train = G_train
            #X_test = G_test
            X = G
            X = model.fit_transform(G)
            pipe = Pipeline([
  #              ('sd', MinMaxScaler()),
                ('clf', clf)
            ])
        #            # K_train = model.fit_transform(X_train)
            # K_test = model.transform(X_test)
        # clf.fit(K_train, y_train)
        # y_pred = clf.predict(K_test)
        # acc = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        #scores = cross_validate(pipe, X=X, y=y, cv=10, scoring=['accuracy', 'f1_macro', 'f1_micro'], n_jobs=-1)
        scores = cross_validate_Kfold_SVM([X], y=y, random_state=42)
        time_took = time.time() - time_s
        results.append({'Dataset':dataset_name,
                        'Model':model_name,
                        'Acc':np.mean(scores[0]),
                        'Acc_std':np.std(scores[0]),
                        'Time':time_took})
        print(results[-1])
    print('~'*20)

res = pd.DataFrame(results)
res['dataset_rank'] = res.groupby(['Dataset'])['Acc'].rank(ascending=False)
print(res.groupby('Model').mean().sort_values('dataset_rank'))
res.to_csv("./res_small_PP_kfold_SVM_nolaplace.csv", index=False)