# Recommender
I started this project looking to implement a simple recommender system
to help me choose films for movie nights (which I still do use it for).
It has since turned into a passion project to learn more about the 
various super-interesting ML techniques powering modern recommenders. 
This repository records and documents my learning experiences, from simple
cosine-distance nearest-neighbor methods, to sophisticated Bayesian Matrix
Factorization techniques.

## Simple Nearest-Neighbor Collaborative Filtering\
These models use item-item NN methods, based on a vector of user ratings (explicit/implicit).
They are simple to implement and interpret, and work well with little data. They are however, 
rather inefficient (O(UI)) to compute, scaling poorly in terms of performance. These techniques
are also sometimes referred to as 'memory-based' methods, and they do not involve any learning,
making them less powerful than ML-based methods.

- [x] cosine distance model
- [x] TF-IDF + cosine model
- [ ] BM25 + cosine model

## Matrix Factorization Collaborative Filtering
These methods try to approximately factorize the user-item matrix, into a set of latent factors
(for both user and items). Explicit methods work with explicitly rated items by users (think star
ratings on IMDB), and treat missing data as missing at random. Implicit methods, on the other hand,
can involve non-explicit user-item interations (e.g. number of listens of an album on spotify), and
treat missing entries as information (can indicate non-preference or non-knowledge).

Shown below is a list of stuff I have worked on, mostly implemented in keras/tf

- [x] Explicit (SVD-like factorization)
- [x] Explicit (Factorization plus bias)
- [x] Implicit (Alternating Least-Squares: http://yifanhu.net/PUB/cf.pdf)
- [x] Implicit (Logistic Matrix Factorization)
- [ ] Implicit (Bayesian Personalized Ranking)

## Bayesian techniques
This is where it gets exciting! I have recently looked up the use of Latent Semantic Analysis (LSA),
traditionally used in NLP, in recommender systems. I have found a huge repository of bayesian methods
in recommender systems, and am looking forward to working on this!

## Example Visualization

Visualization of embeddings extracted from the Movielens dataset:

Dimensionality-reduced embedding plot showing clustering of films from the Harry Potter franchise

![alt text](https://github.com/whong92/recommender/blob/master/notebooks/images/HarryPotter.png "Harry Potter Clustering")

Loose clustering of horror genres

![alt text](https://github.com/whong92/recommender/blob/master/notebooks/images/Horror.png "Harry Potter Clustering")

## Benchmark studies

Coming soon