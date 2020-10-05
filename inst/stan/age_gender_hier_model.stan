//
data {
  int<lower=0> N; // Number of obs
  int<lower=0,upper=1> survived[N];
  vector<lower=0>[N] age;
  int<lower=0> sex; // Number of genders
  int<lower=1, upper=sex> sex_idx[N];

  // Ingest test data to make predictions
  int<lower=0> test_N;
  vector<lower=0>[test_N] test_age;
  //reuse sex variable from above but index over test set
  int<lower=1, upper=sex> test_sex_idx[test_N];
}

parameters {
  real mu;
  vector[sex] alpha;
  vector[sex] beta;
  real<lower=1> sigma;
}

model {
  mu ~ normal(0,10);
  for ( s in 1:sex ){
    beta[s] ~ normal(mu,sigma);
  }
  for( n in 1:N ){
    survived[n] ~ bernoulli_logit( alpha[sex_idx[n]] + age[n] * beta[sex_idx[n]] );
  }
}

generated quantities {
  vector[test_N] y_new;
  int y_rep[N];

  // for ( b in 1:sex ) {
  //   beta[b] = normal_rng(mu,sigma);
  // }

  for ( n in 1:N ) {
    y_rep[n] = bernoulli_logit_rng( alpha[sex_idx[n]] + age[n] * beta[sex_idx[n]] );
  }

  //predictions using test data come from the below process
  for ( i in 1:test_N ){
    y_new[i] = bernoulli_logit_rng( alpha[test_sex_idx[i]] + test_age[i] * beta[test_sex_idx[i]] );
  }
}
