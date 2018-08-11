

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


from gensim.models.doc2vec import TaggedDocument
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
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
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



from sklearn.pipeline import Pipeline
from itertools import product
import multiprocessing as mp
import pandas as pd
import numpy as np


# # Corpus Configuration
# The following configuration determines the input paths as well as the result paths for potentially multiple datasets. The input format was already defines in the introduction of one of the previous [notebooks](1.0_Amazon_corpus_to_pandas.ipynb#Amazon-review-corpus).
# In case you don't define any of the `gs_params` they will be set to one sensible default value.

# In[7]:



# In[6]:


jobs=4

pbt_experiment_data = {
    'sample_sizes': (10000

,),
    'corpora': {
        'amazon' : {
            'input_path': '/root/data/masterarbeit/dataframe/tempfiletokenized.pkl',
        },
           
    },
    'pbt_params': { # This dict defines the parameters to test for
        'trans__size' : [1,300],
        'trans__iter': [1,50],
        'trans__alpha': [0.001, 0.5],
        'trans__negative': [1, 50],
        'trans__window': [1,150,],
        'trans__hs': [0, 1,],
        'trans__sg': [0, 1,],
        'cls__n_neighbors': [5,],
    },
    'param_types': {
        'trans__size' : int,
        'trans__iter': int,
        'trans__alpha': float,
        'trans__negative': int,
        'trans__window': int,
        'trans__hs': int,
        'trans__sg': int,
        'cls__n_neighbors': int,
    },
    'param_std': {
        'trans__size' : 1,
        'trans__iter': 1,
        'trans__alpha': 0.01,
        'trans__negative': 1,
        'trans__window': 1,
        'trans__hs': 1,
        'trans__sg': 1,
        'cls__n_neighbors': 1,
    },
    'param_labels': { # so python names are independent of names in the plots
        'trans__size' : r'd',
        'trans__iter': r'epoch',
        'trans__alpha': r'\alpha',
        'trans__negative': r'ns',
        'trans__window': r'win',
        'trans__hs': r'hs',
        'trans__sg': r'arch',
       'cls__n_neighbors' 
: r'k',        
    }
}



# # Start Grid Search

# In[ ]:


from datetime import datetime
import pickle




# In[8]:


#result = pd.read_pickle('dataframe/20180422-20:03_gs_resultshelo.pkl')
#result['corpora']['amazon']['gs_results']


# In[ ]:


#result


# # Population based training

# In[7]:


from random import sample
from sklearn.model_selection import cross_val_predict, cross_val_score


# In[8]:


class Worker():
    def __init__(self, index, X, y, pipeline, hyperparameters, accuracety_fun, param_types, param_std):
        # position in list
        self.id = index
        
        self.X = X
        self.y = y
        
        # parameter standard deviation used for noise
        self.param_std = param_std
        # parameter type. used as convertor when applying noise
        self.param_types = param_types
        # defining search space (min, max) for each parameter
        self.param_clip = hyperparameters
        
        # creating random hyper-parameters
        self.hyperparameters = {
            param: 
            self.param_types[ param] (
                np.random.uniform( 
                    low=min( values),
                    high=max( values)
                )
            )
            for param, values in hyperparameters.items()
        }
        self.accuracety_fun = accuracety_fun
        
        self.model = None
        self.pipeline = pipeline
        
        
    def get_hyperparameters(self):
        ''' joining parameters together on which we will apply hash function '''
        return ' '.join(
            sorted(
            [param + str(value)
             for param, value in self.hyperparameters.items()]
        ))
        
    def step(self):
        ''' Training model '''
        self.model = self.pipeline
        self.model.set_params( **self.hyperparameters)
        
        score = cross_val_score( self.model, self.X, self.y, cv=5)
        self.accuracety = score.mean()
        self.acc_std = score.std()
        
    def eval(self):
        ''' Evaluating model '''
        return self.accuracety
    
    def exploit(self, population):
        ''' Crossover function between sampled agent and this 
            If sampled agent is better then this, replace this hyper-parameters with his 
            and apply explore function on it.
        '''
        
        current_scores = [{
            "id": agent.id,
            'hyper-parameters': agent.hyperparameters,
            "score": agent.eval()
        } for agent in population]
        
        sampled_agent = sample( current_scores, 1)[0]
        
        if self.accuracety < sampled_agent[ 'score']:
            self.hyperparameters = sampled_agent[ 'hyper-parameters']
            return True
        
        return False

    def explore(self):
        ''' Mutate hyper-parameter where making shore they are correct type and in searched spece '''
        new_hyperparameters = {
            param: 
            np.clip(
                self.param_types[ param](
                    value + np.random.randn() * self.param_std[ param]
                ),
                a_min=min(self.param_clip[ param]),
                a_max=max(self.param_clip[ param])
            )
            for param, value in self.hyperparameters.items()
        }
        
        self.hyperparameters = new_hyperparameters


# In[9]:

print('Starting Optimization')
# Defining random population as initialization for Population based training
def define_population(data, X, y, pipeline, n_population):
    accuracety_fun = lambda y, y_pred: np.mean( np.array( y) == np.array( y_pred))
    return [
        Worker( index, X, y, pipeline, data['pbt_params'], accuracety_fun, data['param_types'], data['param_std'] )
        for index in range( n_population)
    ]


# In[10]:


# Used to convert hyper-parameters from dict of worker to pandas DataFrame
def worker_to_dataframe( worker, df):
    params = worker.hyperparameters.copy()
    params[ 'score mean'] = worker.accuracety
    params[ 'score std'] = worker.acc_std
    worker_df = pd.DataFrame( params, index=[0])
    
    if df is None:
        return worker_df
    return df.append( worker_df)


# In[11]:


# run pbt on sampled size just like in Grid search
def pbt_on_sub_size( X, y, pipeline, h_params, n_steps=10, n_population=5):
    # initalized population just like in Genetic algorihm does
    population = define_population( h_params, X, y, pipeline, n_population)
    
    # df is container for hyper-parameter and result of models
    df = None
    # let's keep track which hyper-parameters where tested
    sub_results = {}

    for iteration in range( n_steps):
        for worker in population:
            if iteration != 0:
                new_params = worker.exploit( population)

                if new_params:
                    worker.explore()
               
            # hash of models hyper-parameters
            hp_hash = hash( worker.get_hyperparameters())
            
            if hp_hash not in sub_results:
                # training model
                worker.step()
                # saving accuracety with hashed hyper-parameters
                sub_results[ hp_hash] = worker.eval()
                
                df = worker_to_dataframe( worker, df)
                
            else:
                # setting model accuracety because we didn't traine it!
                worker.accuracety = sub_results[ hp_hash]
                
            
        bw = sorted(population, key=lambda x: x.eval())[-1]
        print('iteration: ', iteration, ' current acc: ', bw.eval())
    
    return df


def population_based_training(h_parameters, corpus_path, n_steps=50, n_population=10, n_jobs=4, sample_sizes=2):
    txt_clf = Pipeline([
            
            ('trans', W2VTransformer(workers=mp.cpu_count())),
            ('cls',W2VClassifier() )
        ])
    
    results = None
    for sample_size in sample_sizes:
        corpus_adapter = Corpus_adapter(corpus=corpus_path, sample_size=sample_size)
        X = [f['text'] for f in corpus_adapter]
        y = [f['category'] for f in corpus_adapter]

        # runnig pbt for sampled size
        sub_results = pbt_on_sub_size(X, y, txt_clf, h_parameters, n_population=n_population, n_steps=n_steps)
        sub_results[ 'sample size'] = sample_size
        
        # extanding dataframe
        if results is None:
            results = sub_results
        else:
            results = results.append( sub_results)
    
    return results

# In[ ]:


res = population_based_training(pbt_experiment_data, pbt_experiment_data[ 'corpora']['amazon']['input_path'], sample_sizes=pbt_experiment_data['sample_sizes'])
new = res.sort_values('score mean', ascending=False)
new.to_pickle('/root/data/masterarbeit/dataframe/popuresult/popuresultfinal10000new.pkl')
new

# In[ ]:


res.sort_values('score mean', ascending=False)


# # Bayesian Optimization
# The code that is presented in this section is based on the work of [Thomas Huijskens](https://thuijskens.github.io/2016/12/29/bayesian-optimisation/). It implements Bayesian Optimization to find optimal hyperparameter values for the task at hand[1].
# 
# [1] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems (pp. 2951-2959).

# In[8]:


import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize


