# iPMCMC

Python implemenation of iPMCMC: http://proceedings.mlr.press/v48/rainforth16.pdf

*Work in progress*

## Experiments

We chose to implement the exact same models as in the paper at first
where the unobserved states only depend on their previous state, and the observations only depends on the current state. So these are hidden markov models.

### Linear Gaussian State Space Model

The initial state follows a Gaussian distribution centered in mu = (0, 0, 1) with a variance V = 0.1 * Id.

At the time t, the state x_t simply follows a Gaussian distribution, centered on a rotationed version of x_(t-1) (7pi/10, 3pi/10 and pi/20 on the first, second and third dimensions of x_(t-1)), with a variance omega = Id.

The observation model is also a simple Gaussian distribution, so y_t follows this Gaussian centered in Beta * x_t, with Beta an emission matrix with independent columns sampled from a Dirichlet distribution. Its variance Sigma is equal to V.
