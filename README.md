This repository contains the source code and files for the master thesis "Hyperparameter Studies on Word Embedding" by Odinukwe Gerald C. In the master thesis, an experiment was carried out to study the influences of some of the hyperparameters of word2vec on the model accuracy by performing a sentiment anlalysis using the k nearest neighbor classifier. More details are availabe on the paper.

The project was developed using Python 3.5, the .py file can be run from the terminal as a python script while the .ipyb needs to be opened from Jupyter notebooks. The following python libraries are required to be installed for the experiment: Scikit Learn, Matplotlib, Pandas, Gensim, Numpy and Nltk.  

Running this experiment requires a high machine that is greater than 128GB RAM and also greater than 12 CPU cores

The source code files are two ipyb notebooks and four .py files.
The grid.py carries out grid search while bayesian.py carries out Bayesian optimization and the population.py performs population based training. The other .py file traintestsplit.py also performs sentiment analysis without using any hyperparameter optimisation method.
The ExperimentResult.ipyb notebook visualizeses the results for the grid search while the preprocessing.ipyb performs the preprocessing of the dataset.
The output of the grid.py is saved as pickle on the disk of which the content can be opened using the ExperimentResult.ipyb.

