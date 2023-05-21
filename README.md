# Introduction to Machine and Deep Learning Theory
Materials of Lecture course provided in Winter/Spring 2023 in two largest Russia's Technical Universities:
* [Lomonosov Moscow State University](https://www.msu.ru/en/) (Faculty of Mechanics and Mathematics)
* [Moscow Institute of Physics and Technology](https://en.wikipedia.org/wiki/Moscow_Institute_of_Physics_and_Technology) (Master's program "Methods and technologies of artificial intelligence")

# Course overview
Deep learning is a young (approximate date of origin is 2011-2012), but actively developing area of machine learning, which is characterized primarily by the use of neural networks with a large (hence the word “_deep_” in the name) number of layers in their architecture. Initially, deep learning was a predominantly empirical field of knowledge in which new findings were primarily found experimentally. Subsequently, many findings began to receive theoretical justification, and somewhere theory is now even ahead of practice. The course will cover the basic concepts that are used in the theoretical consideration of empirical methods of machine and deep learning - the reasoning behind of loss functions, working with data distributions, theory of generative models, adversarial learning, stability of neural networks, and limit theorems.

# Course content
* Empirical risk and its approximation
  * The basic concepts of measuring the quality of the work of a machine learning algorithm are empirical risk and its approximation. Differentiability. Stochastic gradient descent. Regularization. Probabilistic meaning of loss functions on the example of maximum likelihood and a posteriori probability.
* Basic loss functions. Its evolution based the problem of face recognition
  * The main classification functions of losses are logistic, cross entropy. Entropy and the Gibbs inequality. Functions on distributions. Kullback-Leibler distance. Evolution of loss functions on the example of the problem of face recognition.
* Theoretical justification of adversarial learning methods
  * The mechanism of adversarial learning as a minimax. Derivation of formulas reflecting a practical approach to training. Connection with the Wasserstein metric.
* Adversarial examples and defense against them
  * The surprising effect of the instability of neural networks to the input perturbations. Examples of adversarial perturbations. Basic methods for constructing adversarial examples and defending against them. Classification of adversarial examples. Adversarial Examples implementable in the Real-World.
* Certified Robustness
  * The concepts of certificate and certified robustness. Classical approach using randomized smoothing. Neyman-Pearson Lemma. Curse of dimensionality in computer vision problems.
* Variational Inference
  * Bayes' theorem and posterior probability. Approximation using a parametric family of distributions. Lower bound by ELBO.
* AE, VAE and CVAE
  * Concepts of autoencoder, variational autoencoder, conditional variational autoencoder and their differences. Architectural implementation in practice.
* Markov Chain Monte Carlo
  * The problem of sampling from an empirical space in a (high) multidimensional space. Gibbs, Metropolis and Metropolis-Hastings samplers. Relationship with Langevin dynamics.
* Diffusion Models
  * Forward and reverse process as an analogue of the diffusion process. Derivation of formulas and architectural implementation in practice.
* Limiting theorems for the training process
  * Limiting (existence) theorems for approximation, as well as the dynamics of the convergence of the training process.

# License
**Creative Commons**: BY Attribution-NonCommercial-ShareAlike 4.0 International ([**CC BY-NC-SA**](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode))
