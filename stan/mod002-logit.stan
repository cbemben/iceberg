//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'x' of length 'N'.
data {
  int<lower=0> N;
  vector[N] x; // Age of passenger
  int<lower=0,upper=1> y[N]; // survival outcome
}

// The parameters accepted by the model. Our model
// accepts two parameters 'alpha' and 'beta'.
parameters {
  real alpha;
  real beta;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  y ~ bernoulli_logit( alpha + beta * x );
}

generated quantities {
  int<lower=0,upper=1> y_rep[N];
  for (n in 1:N)
    y_rep[n] = bernoulli_logit_rng( alpha + beta * x[n]);
}

