# Recommender
Various implementations of recommender systems. Currently implemented are:

## Simple Collaborative Filtering
- [x] cosine distance model
- [ ] TF-IDF + cosine model
- [ ] BM25 + cosine model

## Matrix Factorization Collaborative Filtering
- [x] Explicit (SVD-like factorization)
- [x] Explicit (Factorization plus bias)
- [ ] Implicit (Alternating Least-Squares: http://yifanhu.net/PUB/cf.pdf)
- [ ] Implicit (Bayesian Personalized Ranking)

## Example

Visualization of embeddings extracted from the Movielens dataset:

Dimensionality-reduced embedding plot showing clustering of films from the Harry Potter franchise

![alt text](https://github.com/whong92/recommender/blob/master/notebooks/images/HarryPotter.png "Harry Potter Clustering")

Loose clustering of horror genres

![alt text](https://github.com/whong92/recommender/blob/master/notebooks/images/Horror.png "Harry Potter Clustering")
