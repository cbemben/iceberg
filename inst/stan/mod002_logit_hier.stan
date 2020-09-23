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
  int<lower=0> N; // Number of obs
  int<lower=0,upper=1> survival[N];
  vector<lower=0>[N] age; 
  
  int<lower=0> Pclass; // Number of groups
  int<lower=1, upper=Pclass> pclass_idx[N];
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real beta; //coefficient on age
  vector[Pclass] alpha;
  
 // vector[Pclass] mu;        // Passenger class-specific intercepts
  //real<lower=0> sigma_mu;  // sd of passenger class-specific intercepts
  //real phi;               // intercept of model for mu
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  survival ~ bernoulli_logit( alpha[pclass_idx] + age * beta );
}

generated quantities {
  int y_rep[N];
  for ( n in 1:N ) {
    y_rep[n] = bernoulli_logit_rng( alpha[pclass_idx[n]] + age[n] * beta );
  }
}
