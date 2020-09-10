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

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  vector[N] age;
  int<lower=0> Pclass;
  int<lower=1> gIndex[N];
  int<lower=0,upper=1> survival[N];
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector [Pclass] alpha;
  vector [Pclass] beta;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  alpha ~ normal(0,10);
  for ( n in 1:N ) {
    survival[n] ~ bernoulli_logit( alpha[gIndex[n]] + beta[gIndex[n]] * age[n] );
  }
}

generated quantities {
  real y_rep[N];
  for ( n in 1:N ) {
    y_rep[n] = bernoulli_logit_rng( alpha[gIndex[n]] + beta[gIndex[n]] * age[n]);
  }
}
