//
data {
  int<lower=0> N; // Number of obs
  int<lower=0,upper=1> survived[N];
  vector<lower=0>[N] age;
  vector<lower=0,upper=1>[N] class_1;
  vector<lower=0,upper=1>[N] class_2;
  vector<lower=0,upper=1>[N] class_3;
  int<lower=0> sex; // Number of genders
  int<lower=1, upper=sex> sex_idx[N];

  // Ingest test data to make predictions
  int<lower=0> test_N;
  vector<lower=0>[test_N] test_age;
  vector<lower=0,upper=1>[test_N] test_class_1;
  vector<lower=0,upper=1>[test_N] test_class_2;
  vector<lower=0,upper=1>[test_N] test_class_3;
  //reuse sex variable from above but index over test set
  int<lower=1, upper=sex> test_sex_idx[test_N];
}

parameters {
  real beta;
  real beta_class_1;
  real beta_class_2;
  real beta_class_3;
  vector[sex] alpha;
}

model {
  beta ~ normal(0,1);
  beta_class_1 ~ normal(0,1);
  beta_class_2 ~ normal(0,1);
  beta_class_3 ~ normal(0,1);

  survived ~ bernoulli_logit( alpha[sex_idx] + beta * age + beta_class_1 * class_1 + beta_class_2 * class_2 + beta_class_3 * class_3 );
}

generated quantities {
  vector[test_N] y_new;
  int y_rep[N];

  for ( n in 1:N ) {
    y_rep[n] = bernoulli_logit_rng( alpha[sex_idx[n]] + beta * age[n] + beta_class_1 * class_1[n] + beta_class_2 * class_2[n] + beta_class_3 * class_3[n] );
  }

  //predictions using test data come from the below process
  for ( i in 1:test_N ){
    y_new[i] = bernoulli_logit_rng( alpha[test_sex_idx[i]] + beta * test_age[i] + beta_class_1 * test_class_1[i] + beta_class_2 * test_class_2[i] + beta_class_3 * test_class_3[i] );
  }
}
