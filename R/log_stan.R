#' Age gender Bayesian logistic regression with Stan
#'
#' @param age Numeric vector of age values.
#' @param sex_cats integer value of total number of sexes possible (in this case it's 2).
#' @param sex_idx integer value identifying the passenger sex.
#' @param survived integer vector of output values, this is binary.
#' @param test_sex_idx integer value identifying the passenger sex from the test dataset.
#' @param test_age Numeric vector of age values from test dataset.
#' @param ... Arguments passed to `rstan::sampling` (e.g. iter, chains).
#' @return An object of class `stanfit` returned by `rstan::sampling`
#' @export
#'
age_gender_stan <- function(age, sex_cats, sex_idx, survived, test_sex_idx, test_age, ...) {
  standata <- list(N=length(survived),
                   age=age,
                   sex=sex_cats,
                   sex_idx=sex_idx,
                   survived=survived,
                   test_N=length(test_age),
                   test_age=test_age,
                   test_sex_idx=test_sex_idx)
  out <- rstan::sampling(stanmodels$age_gender_model, data = standata, ...)
  return(out)
}
