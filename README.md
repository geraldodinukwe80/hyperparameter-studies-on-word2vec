This repository contains the source code and files for the master thesis "Hyperparameter Studies on Word Embedding" by Odinukwe Gerald C. In the master thesis, an experiment was carried out to study the influences of some of the hyperparameters of word2vec on the model accuracy by performing a sentiment anlalysis usin the k nearest neighbor classifier. More details is availabe on the paper further details, please refer to the paper.


The source code files are two ipyb notebooks and four .py files.
The output of the grid.py is saved as pickle on the disk of which the content can be opened using the ExperimentResult.ipyb
The grid.py carries out grid search while bayesian.py carries out Bayesian optimization and the population.py performs population based training. The other .py file traintestsplit.py also performs sentiment analysis without using any hyperparameter optimisation method.
The ExperimentResult.ipyb notebook visualizeses the results for the grid search whiiel the preprocessing.ipyb performs the preprocessing of the dataset.
Hardware requirements
This requires a high machine that is >128GB RAM, >12 CPU cores


The project was developed using Python 3.5, the .py file can be run from the terminal as a python script while the .ipyb need to be opened from Jupyter notebooks. The following python liberaries are required to be installed for the experiment: Scikit Learn, Matplotlib, Pandas, Gensim, Numpy and Nltk.  

