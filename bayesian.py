
# coding: utf-8

# # Hyperparameter optimization
# 
# This notebook implements the entire processing pipeline to find optimal hyperparameters. It is composed of the following components:
# 1. Text preprocessor that filters and normalizes the input ([`Filter` + `TextPreprocessor`](#Text-Preprocessing))
# 2. An adapter that allows easy iteration over the content of Pandas DataFrames ([`Corpus_adapter`](#Corpus-Adapter))
# 3. The [`Word2Vec Transformer`](#Word2Vec-Transformer) builds a document embedding model and transforms text pieces into their Word2Vec representation.
# 4. The [`Doc2Vec-Classifier`](#Doc2Vec-Classifier) predicts the class/category of a document.
# 5. In the [`Pipeline-and-Hyperparameter-optimization`](#Pipeline-and-Hyperparameter-optimization) section the aforementioned pieces are put together and the processing is carried out.
# 
# You may have noticed that there is a certain interface design style (e.g. `fit()`, `transform()` and `predict()` methods). This is due to the framework provided by [scikit-learn](http://scikit-learn.org/stable/) which allows a simple automation of the hyperparameter


# # Corpus Adapter

# The `Corpus_adapter` class wraps around the pandas dataframe. It creates a sub dataset by randomly picking `sample_size` documents. Since this class implements the `__iter__` method, instances allow simple iteration over the sub dataset.

# In[1]:



from sklearn.model_selection import train_test_split
import pandas as pd

class Corpus_adapter():
    def __init__(self, corpus, sample_size=10000, random_state=42):
        self.df = pd.read_pickle(corpus)
        
        assert sample_size >= 0, 'Sample size must be positive'
        if sample_size >= len(self.df):
            print('sample_size to large. will be set to max val: {}'.format(len(self.df)))
            sample_size = len(self.df)
        
        rnd = np.random.RandomState(random_state)
        self.sample = rnd.choice(self.df.index, sample_size, replace=False)
    
    def __iter__(self):
        for idx in self.sample:
            d = self.df.ix[idx]
            yield {
                    'idx': idx, 
                    'text': d['tokens'], 
                    'category': d['Sentiment']
                }


# # Word2Vec Transformer

# This transformer class implements the methods necessary to be processed by `sklearn.pipeline.Pipeline` (`fit` and `transform`) as well as `sklearn.model_selection.GridSearchCV` (`get_params` and `set_params`).
# It contains the Word2Vec model which is trained by the `fit` method. The `transform`method accepts an array-like object containing text documents that are transformed to their vector representation.
# 
# __Note__: `set_params` and `get_params` have no effect on the model whatsoever. Their purpose is merely to conform to the interface `sklearn.model_selection.GridSearchCV` defines. Changing a model parameter requires re-training. Fortunately `GridSearchCV` instanciates for each configuration a new `D2VTransformer` and calls the `fit` method, which ensures appropriate behavior.

# In[2]:


import os
import gensim
import logging
from gensim.models import Word2Vec
from tqdm import tqdm
from collections import defaultdict, OrderedDict

class W2VTransformer( ):      
        
      
                                             
       
    def __init__(self,**kwargs):
        
        self.kwargs=kwargs
        self.params = OrderedDict({
                'size': 24,
                'window': 10,
                'workers': 1, 
                'min_count': 3, 
                'sg': 0,
                'hs': 0,
                'iter': 5,
                'negative': 10,
                'alpha': .025, 
                'min_alpha': 1e-4, 
                'batch_words': 1000,
                'seed': 42,
            })
        self.params.update(kwargs)
        self.model = None
        
    
    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.Word2Vec
        """
        class IterList():
            def __init__(self, X):
                self.X = X
            def __iter__(self):
                return iter(self.X)
        listIter = IterList(X)
        
        
        
        self.sentences =listIter
        
        self.model = Word2Vec(**self.params )   
        
        self.model.build_vocab(tqdm(self.sentences))
        self.model.train(tqdm(self.sentences),total_examples=self.model.corpus_count, epochs=self.model.epochs)               
        self.model.init_sims() # this initializes the syn0norm array
        return self
        
    
    def transform(self, X):
        self.dim = self.params['size'] 
        return np.array([
        np.mean([self.model[w] for w in words if w in self.model] 
            or [np.zeros(self.dim)], axis=0)
       for words in X])
        
        
    
    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(**params)
        return self


# # Word2Vec Classifier

# This class implements a k-nearest neighbors classifier with cosine similarity distance metric (`predict` method). Furthermore, it implements the `score` method which determines the quality of the prediction with respect to a given dataset.

# In[3]:


class W2VClassifier():
    def __init__(self, **kwargs):
        self.params = {
            'k': 5
        }
        self.params.update(kwargs)
    
    def fit(self, X, y):
        self.data = X
        self.targets = y
    
    def predict(self, X):
        dists = np.dot(X, self.data.transpose())
        best = np.fliplr(np.argsort(dists, axis=1))
        res = []
        for i, bestk in enumerate(best[:, 0:self.params['k']]):
            counter = defaultdict(int)
            for j, idx in enumerate(bestk):
                counter[self.targets[idx]] += dists[i][idx]
            counter = [(cat, val) for cat, val in counter.items()]
            res.append(sorted(counter, key=lambda x: -x[1])[0][0])
        return np.array(res)
    
    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(**params)
        return self




# # Pipeline and Hyperparameter optimization

# In the following section the previously discussed parts are coming together:
# 1. The dataset gets loaded.
# 2. A transformer-classifier pipeline is built.
# 3. The hyperparameter space gets defined.
# 4. The grid search is carried out
# 
# Since this part is crucial, a couple of things should be highlighted. The `Pipeline` class integrates multiple transformation steps with a concluding classification step into one object. The resulting object behaves just like a normal scikit-learn classifier with the exception that it accepts an unprocessed corpus. All the preprocessing is hidden within the `Pipeline` object.
# 
# The hyperpatemeter search (`GridSearchCV`) works on that classifier as well as a set of pre-defined hyperparameters which it is supposed to test for. `GridSearchCV` executes an exhaustive search on the entire parameter space which obviously limits the parameter space that can be searched or rather imposes demands that some machine won't comply with. For situations like these [`RandomizedSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) is more appropriate.
# 
# As the `CV` in `GridSearchCV` indicates, Cross-validation is also performed along the way.

# In[4]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from itertools import product
import multiprocessing as mp
import pandas as pd
import numpy as np

# # Bayesian Optimization
# The code that is presented in this section is based on the work of [Thomas Huijskens](https://thuijskens.github.io/2016/12/29/bayesian-optimisation/). It implements Bayesian Optimization to find optimal hyperparameter values for the task at hand[1].
# 
# [1] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize
print("starting")
def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values of the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return expected_improvement


# In[9]:


import sklearn.gaussian_process as gp
from collections import OrderedDict

def gen_params(param_template, size=None):
    new_params = OrderedDict({})
    for param_name, gen_func in param_template.items():
        new_params[param_name] = gen_func(size)
    return new_params

def param_dict_to_numpy_array(params):
    return np.array([[v for k, v in p.items()] for p in params])

def bayesian_optimisation(n_iters, sample_loss, param_template, n_pre_samples=5,
                          gp_params=None, random_search=10, alpha=1e-5, epsilon=1e-7, verbose=False):
    """ bayesian_optimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        param_template: dict.
            Contains 'param_name': generator_function pairs
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []
    
    for i in range(n_pre_samples):
        new_params = gen_params(param_template)

        x_list.append(new_params)
        y_list.append(sample_loss(new_params))

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)
    if verbose:
        print('beginning optimization')
        
    for n in range(n_iters):
        xp = np.array(param_dict_to_numpy_array(x_list))
        yp = np.array(y_list)
        model.fit(xp, yp)

        # Sample next hyperparameter
        new_params = []
        for i in range(random_search):
            new_params.append(gen_params(param_template))
        x_random = param_dict_to_numpy_array(new_params)
        
        # x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
        ei = expected_improvement(x_random, model, yp, greater_is_better=True, n_params=x_random.shape[1])
        next_sample = new_params[np.argmax(ei)]

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(param_dict_to_numpy_array([next_sample]) - xp) <= epsilon):
            next_sample = gen_params(param_template)

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)
        
        if verbose:
            print('{} -> {}'.format(next_sample, cv_score))

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp


# In[19]:


from sklearn.model_selection import cross_val_score

def sample_loss():
    def inner_loss(params, data=None, target=None):
        params = params.copy()
        clf_param = {'k': 6}
        if 'k' in params:
            clf_param['k'] = params.pop('k')
            
        txt_clf = Pipeline([
               # ('prepro', TextPreprocessor()),
                ('trans', W2VTransformer(workers=mp.cpu_count(), **params)),
                ('cls', W2VClassifier())
            ])
        return cross_val_score(txt_clf, X=data, y=target, cv=5).mean()
    
    corpus_path = '/root/data/masterarbeit/dataframe/tempfiletokenized.pkl'
    corpus_adapter = Corpus_adapter(corpus=corpus_path, sample_size=10000)
    data = [f['text']  for f in corpus_adapter]
    target = [f['category'] for f in corpus_adapter]
    
    return partial(inner_loss, data=data, target=target)


# In[1]:




# In[34]:



# In[ ]:

print("still working")
from functools import partial

param_template = OrderedDict({
    'size': (lambda size: np.random.randint(2, 400, size=size)),
    'window': (lambda size: np.random.randint(1, 150, size=size)),
    'negative': (lambda size: np.random.randint(1, 30, size=size)),        
    'alpha': (lambda size: 10 ** -np.random.uniform(0, 2, size=size)), 
    'iter': (lambda size: np.random.randint(3, 51, size=size)),        
    'sg': (lambda size: np.random.randint(0, 2, size=size)),
    'hs': (lambda size: np.random.randint(0, 2, size=size)),
    'k': (lambda size: np.random.randint(8, 9, size=size))
})

xp, yp = bayesian_optimisation(n_iters=100,
                               sample_loss=sample_loss(), 
                               param_template=param_template,
                               n_pre_samples=10,
                               random_search=10000,
                               verbose=True
                              )


# ### Inspecting the results

# In[ ]:


for ix in list(reversed(yp.argsort()))[:500]:
    print("Accuracy: {} \n {}\n".format(yp[ix], xp[ix]))

