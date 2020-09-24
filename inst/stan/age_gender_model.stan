//
data {
  int<lower=0> N; // Number of obs
  int<lower=0,upper=1> survived[N];
  vector<lower=0>[N] age;

  int<lower=0> sex; // Number of genders
  int<lower=1, upper=sex> sex_idx[N];
}

parameters {
  real beta;
  vector[sex] alpha;
}

model {
  beta ~ normal(0,1);
  survived ~ bernoulli_logit( alpha[sex_idx] + age * beta );
}

generated quantities {
  int y_rep[N];
  for ( n in 1:N ) {
    y_rep[n] = bernoulli_logit_rng( alpha[sex_idx[n]] + age[n] * beta );
  }
}
