# Theoretic Fundamentals of Machine and Deep Learning
Materials of Lecture course that I taught in Winter/Spring 2023 in two largest Russia's Technical Universities:
* [Lomonosov Moscow State University](https://www.msu.ru/en/) (Faculty of Mechanics and Mathematics) - under the name "Introduction to Machine and Deep Learning Theory"
* [Moscow Institute of Physics and Technology](https://en.wikipedia.org/wiki/Moscow_Institute_of_Physics_and_Technology) (Master's program "Methods and technologies of artificial intelligence") - under the name "Introduction to Deep Learning Theory"

## Course overview
Deep learning is a young (approximate date of origin is 2011-2012), but actively developing area of machine learning, which is characterized primarily by the use of neural networks with a large (hence the word “_deep_” in the name) number of layers in their architecture. Initially, deep learning was a predominantly empirical field of knowledge in which new findings were primarily found experimentally. Subsequently, many findings began to receive theoretical justification, and somewhere theory is now even ahead of practice. The course will cover the basic concepts that are used in the theoretical consideration of empirical methods of machine and deep learning - the reasoning behind of loss functions, working with data distributions, theory of generative models, adversarial learning, stability of neural networks, and limit theorems.

## Course content
* **Empirical risk and its approximation**
  * The basic concepts of measuring the quality of the work of a machine learning algorithm are empirical risk and its approximation. Differentiability. Stochastic gradient descent. Regularization. Probabilistic meaning of loss functions on the example of maximum likelihood and a posteriori probability.
* **Basic loss functions. Their evolution based on the problem of face recognition**
  * The main classification functions of losses are logistic, cross entropy. Entropy and the Gibbs inequality. Functions on distributions. Kullback-Leibler distance. Neural Collapse. Evolution of loss functions on the example of the problem of face recognition.
* **Theoretical justification of adversarial learning methods**
  * The mechanism of adversarial learning as a minimax. Derivation of formulas reflecting a practical approach to training. Connection with the Wasserstein metric.
* **Variational Inference**
  * Bayes' theorem and posterior probability. Approximation using a parametric family of distributions. Lower bound by ELBO. Bayesian Neural Networks.
* **AE, VAE and CVAE**
  * Concepts of autoencoder, variational autoencoder, conditional variational autoencoder and their differences. Architectural implementation in practice.
* **Markov Chain Monte Carlo**
  * The problem of sampling from an empirical space in a (high) multidimensional space. Equations of detailed balance. Gibbs, Metropolis and Metropolis-Hastings samplers. Relationship with Langevin dynamics. Adjustment by Metropolis.
* **Diffusion Models**
  * Forward and reverse process as an analogue of the diffusion process. Derivation of formulas and architectural implementation in practice. 3 interpretations of diffusion models. Classifier (-free) Guidance.
* **Adversarial examples and defense against them**
  * The surprising effect of the instability of neural networks to the input perturbations. Examples of adversarial perturbations. Basic methods for constructing adversarial examples and defending against them. Classification of adversarial examples. Adversarial Examples implementable in the Real-World and their common features. 
* **Certified Robustness**
  * The concepts of certificate and certified robustness. Classical approach using randomized smoothing. Neyman-Pearson Lemma. Smoothing distribution vs norm of perturbation. Curse of dimensionality in computer vision problems. Semantic Traansformations.
* **Limiting theorems for the training process**
  * Limiting (existence) theorems for approximation, as well as the dynamics of the convergence of the training process.

## <a name="program" /> Course program
| N        | Lecture (in English)          | Desctription                                            | Video (in Russian)            |
| ------------- | ------------- | -------------                                      | -------------    |        
| 01            | [Emprirical Risk and Loss](/lectures/lecture01-ER_Loss.pdf)    | Empirical Risk and its Approximation. Loss Function. (Stochastic) Gradient Descent. MLE and MAP. Kullback-Leibler divergence and Cross Entropy |  [record01](https://www.youtube.com/watch?v=vBgo_T7V5hE)   |
| 02            | [Representation Learning and FaceID Losses](/lectures/lecture02-ReprLearn_FaceID_Loss.pdf)    | Task of Representation Learning. Neural Collapse. FaceID: Evolution of Loss Function. Representation Learning. SoftMax-, contrastive- and angular-based losses |  [record02](https://www.youtube.com/watch?v=4dwmNbMqcwg)  |
| 03            | [GANs](/lectures/lecture03-GAN.pdf)    | Discriminate vs Generative models. Generative Adversarial Networks. Deep Convolutional GAN. Wasserstein GAN. Gradient-Penalty WGAN. Conditional GAN. |  [record03](https://www.youtube.com/watch?v=qb-4TQIUrzY)  |
| 04            | [Bayes, VI, AE, VAE, cVAE, BNN](/lectures/lecture04-VI_AE_VAE_CVAE.pdf)    | Bayesian Inference, Bayesian Neural Network, Variational Inference, Autoencoder, Variational Autoencoder, Conditional Variational Autoencoder |  [record04](https://www.youtube.com/watch?v=Wf-Hm0SzP5s)  |
| 05            | [Markov Chain Monte Carlo](/lectures/lecture05-MCMC.pdf)    | Recap of Markov Chains. Markov Chain Monte Carlo. Gibbs sampler. Metropolis-Hastings sampler. Langevin dynamics and Metropolis-Adjusted Langevin. Stochastic Gradient Langevin Dynamics |  [record05](https://www.youtube.com/watch?v=FzXEP_JHTgw)  |
| 06            | [Diffusion Models](/lectures/lecture06-Diffusion.pdf)    | Recap of Variational Autoencoder. Markovian Hierarchical VAE. Diffusion models: Variational Diffusion Models, Diffusion Denoising Probabilistic Models, Diffusion Denoising Implicit Models, Classifier and classifier-free guidance, 3 interpretations |  [record06](https://www.youtube.com/watch?v=zeYZfeuvxDk)  |
| 07            | [Adversarial Robustness I: Digital Domain](/lectures/lecture07-AdvRob_I_Digital.pdf)    | Adversarial Robustness I: Great Success of CNNs, Robustness Phenomenon, Taxonomy of Adversarial Attacks, l_p norms, Digital Domain, Fast Gradient Sign Method and its variants, Universal Attacks, Top-k Attacks, l_0 attacks |  [record07](https://www.youtube.com/watch?v=iWjErgoxJuo)  |
| 08            | [Adversarial Robustness II: Real World](/lectures/lecture08-AdvRob_II_Real.pdf)    | Adversarial Robustness II: Adversarial examples in real world, Adversarial attack on Face detection and Face ID systems, Defense from adversarial examples in real world, Black-box Face restoration |  [record08](https://www.youtube.com/watch?v=aQe5a6rLaF8)  |
| 09            | [Certified Robustness I: Randomized Smoothing](/lectures/lecture09-CertRob_I_RS.pdf)    | Certified Robustness I: definitions of Certified Robustness, connection to Lipschitzness, Randomized Smoothing and its variants |  [record09](https://www.youtube.com/watch?v=Rg4OJXQE3K4)  |
| 10            | [Certified Robustness II: High Dimensions and Semantic Transformations](/lectures/lecture10-CertRob_II_HighDim.pdf)    | Certified Robustness II: recap of Certified Robustness, Ablations on base classifier/norm of perturbation/smoothing distribution, Certification in High Dimensional case, Certification of Semantic Perturbations, Application to different Computer Vision tasks |  [record10](https://www.youtube.com/watch?v=kuV1_YFpGo0)  |
| 11            | [Neural Tangent Kernel](/lectures/lecture11-NTK.pdf)    | Neural Tangent Kernel: Lazy regime of training, GD as PDE, NTK and CNTK, NTK convergence rates |  [record11](https://www.youtube.com/watch?v=cN4emH7_EIM)  | 

## <a name="lit" /> Bibliography
1. [Machine Learning Lecture Course](http://www.machinelearning.ru/wiki/index.php?title=Машинное_обучение_%28курс_лекций%2C_К.В.Воронцов%29) on http://www.machinelearning.ru from Vorontsov K.V.
2. Hastie, T. and Tibshirani, R. and Friedman, J. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf), 2nd edition, Springer, 2009.
3. Bishop, C.M. [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), Springer, 2006.
4. Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning. Vol. 1. Cambridge: MIT press, 2016.
5. Matus Telgarsky, Deep learning theory lecture [notes](https://mjt.cs.illinois.edu/dlt/index.pdf), 2021
6. Sanjeev Arora et al., Theory of Deep learning book [draft](https://www.dropbox.com/s/smkp4vasbiszhw4/DLbook.pdf?dl=0), 2020

## <a name="links" /> Useful links 
### Introduction to machine learning
* Homemade Machine Learning: [github repo](https://github.com/trekhleb/homemade-machine-learning)
* Machine learning: [Course](https://www.coursera.org/learn/machine-learning) by Andrew Ng on the site https://www.coursera.org

### Theoretic Courses
* Foundations of Deep Learning: [Course](https://uclaml.github.io/CS269-Spring2021/) at UCLA
* Deep learning theory: [Course](https://mjt.cs.illinois.edu/dlt/) at UIUC
* Theoretical Deep Learning: [Course](https://www.cs.princeton.edu/courses/archive/fall19/cos597B/) at Princeton

# License
**Creative Commons**: BY Attribution-NonCommercial-ShareAlike 4.0 International ([**CC BY-NC-SA**](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode))
