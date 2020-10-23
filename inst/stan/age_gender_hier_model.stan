//
data {
  int<lower=0> N; // Number of obs
  int<lower=0,upper=1> survived[N];
  vector<lower=0>[N] age;
  vector<lower=1>[N] pclass;
  int<lower=0> sex; // Number of genders
  int<lower=1, upper=sex> sex_idx[N];

  // Ingest test data to make predictions
  int<lower=0> test_N;
  vector<lower=0>[test_N] test_age;
  vector<lower=1>[test_N] test_pclass;
  //reuse sex variable from above but index over test set
  int<lower=1, upper=sex> test_sex_idx[test_N];
}

parameters {
  real beta;
  real beta_class;
  vector[sex] alpha;
}

model {
  beta ~ normal(0,1);
  beta_class ~ normal(0,1);

  survived ~ bernoulli_logit( alpha[sex_idx] + beta * age + beta_class * pclass );
}

generated quantities {
  vector[test_N] y_new;
  int y_rep[N];

  for ( n in 1:N ) {
    y_rep[n] = bernoulli_logit_rng( alpha[sex_idx[n]] + beta * age[n] + beta_class * pclass[n] );
  }

  //predictions using test data come from the below process
  for ( i in 1:test_N ){
    y_new[i] = bernoulli_logit_rng( alpha[test_sex_idx[i]] + beta * test_age[i] + beta_class * test_pclass[i] );
  }
}
