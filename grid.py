

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


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from itertools import product
import multiprocessing as mp
import pandas as pd
import numpy as np

def run_parametersearch(gs_params, corpus_path, jobs=4, sample_sizes=(int(1e4), int(1e3))):
    txt_clf = Pipeline([
            
            ('trans', W2VTransformer(workers=mp.cpu_count())),
            ('cls', W2VClassifier())
        ])

    res = {}
    for sample_size in sample_sizes:
        corpus_adapter = Corpus_adapter(corpus=corpus_path, sample_size=sample_size)
        gs_clf = GridSearchCV(txt_clf, gs_params, n_jobs=jobs, cv=5, verbose=1, return_train_score=False)

        X_train = [f['text'] for f in corpus_adapter]
        y_train = [f['category'] for f in corpus_adapter]
        res[sample_size] = gs_clf.fit(X_train, y_train)
    return res


# In[5]:


def gs_to_df(gs_res):
    '''
    gs_to_df gathers the results of a gridsearch run into a pandas dataframe.
    gs_res is a dict who's keys are the sample sizes and who's values are GridSearchCV instances.
    '''
    df = None
    for sample_size, gs in gs_res.items():
        cv_report = gs.cv_results_
        
        mean = cv_report['mean_test_score']
        std = cv_report['std_test_score']
        params = cv_report['params']
        
        mean = pd.Series(mean, name='mean')
        std = pd.Series(std, name='std')
        sample_size = pd.DataFrame([sample_size]*len(params), columns=['sample_size'], dtype=int)
        params = pd.DataFrame((pd.Series(p) for p in params))
        
        params = params.merge(pd.DataFrame(sample_size), left_index=True, right_index=True)
        params = params.merge(pd.DataFrame(mean), left_index=True, right_index=True)
        params = params.merge(pd.DataFrame(std), left_index=True, right_index=True)
        
        if df is None:
            df = params
        else:
            df = df.append(params)
    df = df.reset_index()
    return df


# # Corpus Configuration
# The following configuration determines the input paths as well as the result paths for potentially multiple datasets. The input format was already defines in the introduction of one of the previous [notebooks](1.0_Amazon_corpus_to_pandas.ipynb#Amazon-review-corpus).
# In case you don't define any of the `gs_params` they will be set to one sensible default value.

# In[7]:


jobs=4

experiment_data = {
    'sample_sizes': (10000,),
    'corpora': {
        'amazon' : {
            'input_path': '/root/data/masterarbeit/dataframe/tempfiletokenized.pkl',
        },
           
    },
    'gs_params': { # This dict defines the parameters to test for
        'trans__size' : (64,100,1000,),
        'trans__iter': (1,20,30,),
        'trans__alpha': (.10,.01,),
        'trans__negative': (20,50,),
        'trans__window': (5,50,),
        'trans__hs': (1,0,),
        'trans__sg': (1,),
        'cls__n_neighbors': (5,),
    },
    'param_labels': { # so python names are independent of names in the plots
        'trans__size' : r'd',
        'trans__iter': r'epoch',
        'trans__alpha': r'\alpha',
        'trans__negative': r'ns',
        'trans__window': r'win',
        'trans__hs': r'hs',
        'trans__sg': r'arch',
        'cls__n_neighbors': r'k',        
    }
}


# # Start Grid Search

# In[ ]:


from datetime import datetime
import pickle

for corpus, config in experiment_data['corpora'].items():
    print('Processing {} corpus'.format(corpus))
    results = run_parametersearch(
        experiment_data['gs_params'], 
        config['input_path'], 
        jobs=jobs, 
        sample_sizes=experiment_data['sample_sizes']
    )
    df = gs_to_df(results)
    experiment_data['corpora'][corpus]['gs_results'] = df 

# serialize the results of this run
now = datetime.now()
with open('/root/data/masterarbeit/dataframe/gridresult/{}_gs_resultstateofart.pkl'.format(now.strftime("%Y%m%d-%H:%M")), 'wb') as fh:
    pickle.dump(experiment_data, fh)


# In[8]:


#result = pd.read_pickle('dataframe/20180422-20:03_gs_resultshelo.pkl')
#result['corpora']['amazon']['gs_results'
