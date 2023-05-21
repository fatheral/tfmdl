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
* Variational Inference
  * Bayes' theorem and posterior probability. Approximation using a parametric family of distributions. Lower bound by ELBO.
* AE, VAE and CVAE
  * Concepts of autoencoder, variational autoencoder, conditional variational autoencoder and their differences. Architectural implementation in practice.
* Markov Chain Monte Carlo
  * The problem of sampling from an empirical space in a (high) multidimensional space. Gibbs, Metropolis and Metropolis-Hastings samplers. Relationship with Langevin dynamics.
* Diffusion Models
  * Forward and reverse process as an analogue of the diffusion process. Derivation of formulas and architectural implementation in practice.
* Adversarial examples and defense against them
  * The surprising effect of the instability of neural networks to the input perturbations. Examples of adversarial perturbations. Basic methods for constructing adversarial examples and defending against them. Classification of adversarial examples. Adversarial Examples implementable in the Real-World.
* Certified Robustness
  * The concepts of certificate and certified robustness. Classical approach using randomized smoothing. Neyman-Pearson Lemma. Curse of dimensionality in computer vision problems.
* Limiting theorems for the training process
  * Limiting (existence) theorems for approximation, as well as the dynamics of the convergence of the training process.

## <a name="program" /> Course program
| N        | Lecture (in English)          | Desctription                                            | Video (in Russian)            |
| ------------- | ------------- | -------------                                      | -------------    |        
| 01            | [Emprirical Risk and Loss](/lectures/MM_lecture01-ER_Loss.pdf)    | Empirical Risk and its Approximation. Loss Function. (Stochastic) Gradient Descent. MLE and MAP. Kullback-Leibler divergence and Cross Entropy |  [record01](https://www.youtube.com/watch?v=vBgo_T7V5hE)   |
| 02            | [Representation Learning and FaceID Losses](/lectures/MM_lecture02-FaceID_Loss.pdf)    | Task of Representation Learning. Neural Collapse. FaceID: Evolution of Loss Function. Representation Learning. SoftMax-, contrastive- and angular-based losses |  [record02](https://www.youtube.com/watch?v=4dwmNbMqcwg)  |
| 03            | [GANs](/lectures/MM_lecture03-GAN.pdf)    | Discriminate vs Generative models. Generative Adversarial Networks. Deep Convolutional GAN. Wasserstein GAN. Gradient-Penalty WGAN. Conditional GAN. |  [record03](https://www.youtube.com/watch?v=qb-4TQIUrzY)  |
| 04            | [Bayes, VI, AE, VAE, cVAE, BNN](/lectures/MM_lecture04-VI_AE_VAE_CVAE.pdf)    | Bayesian Inference, Bayesian Neural Network, Variational Inference, Autoencoder, Variational Autoencoder, Conditional Variational Autoencoder |  [record04](https://www.youtube.com/watch?v=Wf-Hm0SzP5s)  |
| 05            | [Markov Chain Monte Carlo](/lectures/MM_lecture05-MCMC.pdf)    | Recap of Markov Chains. Markov Chain Monte Carlo. Gibbs sampler. Metropolis-Hastings sampler. Langevin dynamics and Metropolis-Adjusted Langevin. Stochastic Gradient Langevin Dynamics |  [record05](https://www.youtube.com/watch?v=FzXEP_JHTgw)  |
| 06            | [Diffusion Models](/lectures/MM_lecture06-Diffusion.pdf)    | Recap of Variational Autoencoder. Markovian Hierarchical VAE. Diffusion models: Variational Diffusion Models, Diffusion Denoising Probabilistic Models, Diffusion Denoising Implicit Models, Classifier and classifier-free guidance, 3 interpretations |  [record06](https://www.youtube.com/watch?v=zeYZfeuvxDk)  |
| 07            | [Adversarial Robustness I: Digital Domain](/lectures/MM_lecture07-AdvRob_I_Digital.pdf)    | Adversarial Robustness I: Great Success of CNNs, Robustness Phenomenon, Taxonomy of Adversarial Attacks, l_p norms, Digital Domain, Fast Gradient Sign Method and its variants, Universal Attacks, Top-k Attacks, l_0 attacks |  [record07](https://www.youtube.com/watch?v=iWjErgoxJuo)  |
| 08            | [Adversarial Robustness II: Real World](/lectures/MM_lecture08-AdvRob_II_Real.pdf)    | Adversarial Robustness II: Adversarial examples in real world, Adversarial attack on Face detection and Face ID systems, Defense from adversarial examples in real world, Black-box Face restoration |  [record08](https://www.youtube.com/watch?v=aQe5a6rLaF8)  |
| 09            | [Certified Robustness I: Randomized Smoothing](/lectures/MM_lecture09-CertRob_I_RS.pdf)    | Certified Robustness I: definitions of Certified Robustness, connection to Lipschitzness, Randomized Smoothing and its variants |  [record09](https://www.youtube.com/watch?v=Rg4OJXQE3K4)  |
| 10            | [Certified Robustness II: High Dimensions and Semantic Transformations](/lectures/MM_lecture10-CertRob_II_HighDim.pdf)    | Certified Robustness II: recap of Certified Robustness, Ablations on base classifier/norm of perturbation/smoothing distribution, Certification in High Dimensional case, Certification of Semantic Perturbations, Application to different Computer Vision tasks |  [record10](https://www.youtube.com/watch?v=kuV1_YFpGo0)  |
| 11            | [Neural Tangent Kernel](/lectures/MM_lecture11-NTK.pdf)    | Neural Tangent Kernel: Lazy regime of training, GD as PDE, NTK and CNTK, NTK convergence rates |  [record11](https://www.youtube.com/watch?v=cN4emH7_EIM)  | 

# License
**Creative Commons**: BY Attribution-NonCommercial-ShareAlike 4.0 International ([**CC BY-NC-SA**](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode))
