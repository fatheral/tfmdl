# Theoretic Fundamentals of Machine and Deep Learning

## Student's Projects Proposals
1. Investigate Neural Collapse on different datasets (MNIST, Omniglot, LFW, ...)
2. Make a comparison study of angular-based losses vs metric-based ones on different datasets (MNIST, Omniglot, LFW, ...)
3. Think of evaluation metric for GAN solution (aside from [Inception Score](https://en.wikipedia.org/wiki/Inception_score) / [Frechet Inception Distance](https://en.wikipedia.org/wiki/Fréchet_inception_distance)) and make a coparison study of this metric for different GAN solution: vanilla GAN, WGAN, WGAN-GP
4. Implement and analyze the BNN recognition results using different priors for weights (Uniform, Gaussian, Laplace) on different datasets (MNIST, Omniglot, LFW, ...)
    1. Do it with Variational Inference
    2. Do it with MCMC
5. Explore the Diffusion generation quality vs number of steps on different datasets (MNIST, Omniglot, LFW, ...)
    1. Do it with unconditional generation
    2. Do it with classifier(-free) guidance
    3. Explore different strategies of $\alpha$ ($\beta$) decrease schedule
6. Make a quantitave and qualitative analysis of different $l_0/l_1/l_2/l_{\infty}$-based Adversarial Attacks (success rate, number of iterations, etc) on different datasets (MNIST, Omniglot, LFW, ...)
    1. Do it for the Universal Adversarial Attack as well
    2. Compare the transferability for different NN architectures (LeNet, VGG, ResNet, etc)
7. Create a real-world attack demo for any detection/recognition system
