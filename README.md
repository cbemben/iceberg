The Titanic Kaggle Challenge
================
Chris Bemben
9/5/2020

## Overview

Details of this Kaggle challenge can be found
[here](https://www.kaggle.com/c/titanic), the challenge is to accurately
predict whether a passenger survived the shipwreck or not. The response
variable is binary and since a passenger cannot partially survive the
response variable will be either 1 or 0. This document will use the
[Stan](https://mc-stan.org/) programming language and logistic
regression to attack the challenge.

``` r
library(rstan)
library(rstanarm)
library(magrittr)
library(ggplot2)
library(titanic)
library(bayesplot)

train <- titanic::titanic_train

train_idx <- sample(nrow(train), nrow(train)*0.8)
test_idx <- setdiff(seq_len(nrow(train)), train_idx)

str(train)
```

    ## 'data.frame':    891 obs. of  12 variables:
    ##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
    ##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
    ##  $ Name       : Factor w/ 891 levels "Abbing, Mr. Anthony",..: 109 191 358 277 16 559 520 629 417 581 ...
    ##  $ Sex        : Factor w/ 2 levels "female","male": 2 1 1 1 2 2 2 2 1 1 ...
    ##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
    ##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
    ##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
    ##  $ Ticket     : Factor w/ 681 levels "110152","110413",..: 524 597 670 50 473 276 86 396 345 133 ...
    ##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
    ##  $ Cabin      : Factor w/ 148 levels "","A10","A14",..: 1 83 1 57 1 1 131 1 1 1 ...
    ##  $ Embarked   : Factor w/ 4 levels "","C","Q","S": 4 2 4 4 4 3 4 4 4 2 ...

It’s a common anecdote that women and children were the first passengers
saved so the first model will only use age and gender as predictors.
Since there is a substantial number of missing age values, nulls will be
imputed, see the `Impute Passenger Age` function for the details of the
imputation methodology or the `Exploratory Data Analysis` vignette to
see the before after.

``` r
train <- titanic::impute_passenger_age(train)
```

The `Stan` program fits the model but also accepts a test dataset which
will be used to make predictions. For more details on this approach see
the `Stan`
[manual](https://mc-stan.org/docs/2_24/stan-users-guide/prediction-forecasting-and-backcasting.html).

``` r
simple_model <- titanic::age_gender_stan(
                                          #N=nrow(train[train_idx,]),
                                          age=train[train_idx,'Age'],
                                          sex=2,
                                          sex_idx=as.integer(train[train_idx,'Sex']),
                                          survived=train[train_idx,'Survived'],
                                          #test_N=nrow(train[test_idx,]),
                                          test_age=train[test_idx,'Age'],
                                          test_sex_idx=as.integer(train[test_idx,'Sex']), seed=1234)
```

    ## 
    ## SAMPLING FOR MODEL 'age_gender_model' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 9.9e-05 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.99 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 1.20515 seconds (Warm-up)
    ## Chain 1:                0.54697 seconds (Sampling)
    ## Chain 1:                1.75212 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'age_gender_model' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 5.2e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.52 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 2: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 2: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 2: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 2: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 1.2332 seconds (Warm-up)
    ## Chain 2:                0.583945 seconds (Sampling)
    ## Chain 2:                1.81714 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'age_gender_model' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 5.3e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.53 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 3: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 3: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 3: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 3: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 1.23919 seconds (Warm-up)
    ## Chain 3:                0.572612 seconds (Sampling)
    ## Chain 3:                1.8118 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'age_gender_model' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 6.6e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.66 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 4: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 4: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 4: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 4: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 1.39341 seconds (Warm-up)
    ## Chain 4:                0.624158 seconds (Sampling)
    ## Chain 4:                2.01757 seconds (Total)
    ## Chain 4:

``` r
#print(simple_model)
```

The model estimates an intercept for each `Sex` separately but shares
the coefficient on age. The prior on `beta` is a standard normal
distribution centered at 0 with a standard deviation of 1.

``` r
print(simple_model, pars=c('alpha','beta'))
```

    ## Inference for Stan model: age_gender_model.
    ## 4 chains, each with iter=2000; warmup=1000; thin=1; 
    ## post-warmup draws per chain=1000, total post-warmup draws=4000.
    ## 
    ##           mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
    ## alpha[1]  1.02    0.01 0.24  0.56  0.86  1.02  1.18  1.49   980 1.00
    ## alpha[2] -1.40    0.01 0.23 -1.86 -1.56 -1.40 -1.25 -0.96   850 1.01
    ## beta      0.00    0.00 0.01 -0.02 -0.01  0.00  0.00  0.01   745 1.01
    ## 
    ## Samples were drawn using NUTS(diag_e) at Tue Sep 29 16:57:52 2020.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).

Gender has a big impact on survival.

``` r
samp <- rstan::extract(simple_model)

mcmc_areas(
  simple_model, 
  pars = c("alpha[1]","alpha[2]"),
  prob = 0.8, # 80% intervals
  prob_outer = 0.99, # 99%
  point_est = "mean"
)
```

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
post_pred_df <- as.data.frame(samp$y_new)
bayesplot::ppc_stat(y = train[test_idx,"Survived"], yrep = as.matrix(post_pred_df), stat = mean)
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](README_files/figure-gfm/unnamed-chunk-6-1.png)<!-- --> The plot
above is comparing the predictions to the actual mean of the test
dataset. Overall the model does a good job of estimating the survival
rate but individual predictions are still in need of review and for the
Kaggle challenge we care about accuracy.

``` r
mean(apply(samp$y_new, 2, median) %>% round(0) == train[test_idx,'Survived'])
```

    ## [1] 0.8044693

Accuracy against a test set is about 0.8044693 percent. There seems to
be more to the story we need to consider to improve accuracy to a
reasonable level.

## Fit Logistic Regression

<!-- Age is only explanatory variable at this time. -->

<!-- ```{r, results="hide", message=FALSE} -->

<!-- age_logit_mod <- titanic::stanmodels$mod002_logit -->

<!-- ``` -->

<!-- ```{r} -->

<!-- age_logit_fit <- sampling( -->

<!--   age_logit_mod, -->

<!--   data = list( N = nrow(train),  -->

<!--                x = train$Age,  -->

<!--                y = train$Survived,  -->

<!--                N_new = nrow(test), -->

<!--                cores = 4, -->

<!--                x_new=test$Age), -->

<!--   seed = 111 -->

<!-- ) -->

<!-- ``` -->

<!-- ```{r} -->

<!-- age_logit_ext <- rstan::extract(age_logit_fit) -->

<!-- par(mfrow = c(3,1)) -->

<!-- hist(age_logit_ext$alpha, main = 'alpha param') -->

<!-- hist(age_logit_ext$beta, main = 'beta param') -->

<!-- hist(age_logit_ext$y_rep, main = 'simulated survival') -->

<!-- ``` -->

<!-- ```{r} -->

<!-- posterior <- as.array(age_logit_fit) -->

<!-- bayesplot::mcmc_scatter(posterior, pars=c("alpha","beta")) -->

<!-- ``` -->

<!-- ```{r} -->

<!-- bayesplot::mcmc_intervals(posterior, pars=c("alpha","beta")) -->

<!-- ``` -->

<!-- ```{r, message=FALSE} -->

<!-- #https://discourse.mc-stan.org/t/posterior-prediction-from-logit-regression/12217/2 -->

<!-- postDF <- as.data.frame(age_logit_ext$y_new) -->

<!-- bayesplot::ppc_stat(y = as.integer(train$Survived[1:418]), yrep = as.matrix(postDF), stat = mean) -->

<!-- ``` -->

<!-- ```{r, message=FALSE} -->

<!-- bayesplot::ppc_stat_grouped(y = as.integer(train$Survived[1:418]), yrep = as.matrix(postDF), stat = mean, group = train$Pclass[1:418]) -->

<!-- ``` -->

<!-- The passenger class shows heterogeneitiy across passenger class. However, the model assumes the same survival rate across classes. -->

<!-- ```{r, results="hide", message=FALSE} -->

<!-- age_logit_hier_mod <- stanmodels$mod002_logit_hier -->

<!-- ``` -->

<!-- ```{r} -->

<!-- age_logit_hier_fit <- sampling( -->

<!--               age_logit_hier_mod, -->

<!--               data = list( N = nrow(train), -->

<!--                            age = train$Age, -->

<!--                            Pclass = 3, -->

<!--                            pclass_idx = train$Pclass, -->

<!--                            survival = train$Survived ), -->

<!--               seed = 112, -->

<!--               cores = 4) -->

<!-- ``` -->

<!-- The survival rate is similar to the actual data. -->

<!-- ```{r} -->

<!-- print(age_logit_hier_fit, pars = c('alpha','beta')) -->

<!-- ``` -->

<!-- ```{r} -->

<!-- age_logit_hier_ext <- rstan::extract(age_logit_hier_fit) -->

<!-- y_rep <- as.matrix(age_logit_hier_fit, pars = "y_rep") -->

<!-- ppc_stat_grouped(age_logit_hier_ext, -->

<!--                  y = train$Survived, -->

<!--                  yrep = y_rep[1:891,], -->

<!--                  group = train$Pclass, -->

<!--                  stat = "mean", -->

<!--                  binwidth = 0.009) -->

<!-- ``` -->

<!-- Predict with new data -->

<!-- ```{r} -->

<!-- beta_post <- age_logit_hier_ext$beta -->

<!-- # Function for simulating y based on new x -->

<!-- gen_quant_r <- function(age, Pclass) { -->

<!--   alpha_post<- age_logit_hier_ext$alpha[,Pclass] #get intercept for class of passenger -->

<!--   lin_comb <- sample(alpha_post, size = length(age)) + age*sample(beta_post, size = length(age)) -->

<!--   prob <- 1/(1 + exp(-lin_comb)) #inverse of logit link function -->

<!--   out <- rbinom(length(age), 1, prob) -->

<!--   return(out) -->

<!-- } -->

<!-- y_hat_tr <- gen_quant_r(train$Age, test$Pclass) -->

<!-- mean(y_hat_tr == train$Survived) -->

<!-- ``` -->

<!-- generate predictions on the test data -->

<!-- ```{r} -->

<!-- #y_hat <- gen_quant_r(test$Age, test$Pclass) -->

<!-- # Accuracy -->

<!-- #pred_df <- data.frame(PassengerId = test$PassengerId, Survived=y_hat) -->

<!-- #write.csv(pred_df, file = "~/projectrepos/titanic/data/predict_20200918.csv", row.names = F) -->

<!-- ``` -->
