<h1 align="center">
  Temporal Domain Generalization
</h1>

Given a sequence of time-indexed datasets D_1,D_2,…,D_K, all corresponds to the same prediction task. Over time, the underlying data distribution may change, so each dataset may look different from the previous ones.
Now, a very limited number of labeled samples from a new dataset, D_{K+1} are provided. The goal is to design a strategy that leverages both the historical datasets and the small sample from D_{K+1} to build a machine learning model that performs well on a held-out test set D_{K+1}.
How can we make effective use of the data from the past while adapting to the distribution shift at data-set K+1?

## Method: Kalman Filter
