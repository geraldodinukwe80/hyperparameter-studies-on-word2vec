import pandas as pd
print ("starting to Read")
data = pd.read_pickle("/root/data/masterarbeit/dataframe/tempfiletokenized.pkl")
 
print ("Read Done")

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
                'size': 64,
                'window': 5,
                'workers': 1,  
                'sg': 1,
                'hs': 1,
                'iter': 30,
                'negative': 20,
                'alpha': .10, 
                
            })
        self.params.update(kwargs)
        self.model = None
    
    def fit(self, X, y=None):
        #self.X = X
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
        
        self.model = Word2Vec(**self.params)   
        
        self.model.build_vocab(tqdm(self.sentences))
        self.model.train(tqdm(self.sentences),total_examples=self.model.corpus_count, epochs=self.model.epochs)               
        self.model.init_sims()
        
        return self
        
    #def fit(self,x, y=None):
        #return self

    def transform(self, X):
        #self.dim = self.params['size']
        self.dim = self.model.vector_size 
        return np.array([
        np.mean([self.model[w] for w in words if w in self.model] 
            or [np.zeros(self.dim)], axis=0)
       for words in X])
        
        
    
    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(**params)
        return self



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



from sklearn.model_selection import train_test_split
import numpy as np
df =data
x_train, x_test, y_train, y_test = train_test_split(np.array(df.tokens),
                                                    np.array(df.Sentiment), test_size=0.3)


from sklearn.pipeline import Pipeline
import multiprocessing as mp
txt_clf = Pipeline([
            
            ('trans', W2VTransformer(workers=mp.cpu_count())),
            ('cls', W2VClassifier())
        ])

print ("Fitting")

txt_clf.fit(x_train, y_train)

print ("Predicting")

pred = txt_clf.predict(x_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,pred))
