#' Interface to Stan Models
#' 
#' These functions are the interface to the logistic regression models fit using stan.
#' 
#' @name stan-models
#' 
#' @param age Numeric vector of age values.
#' @param pclass integer value of the passengers class status.
#' @param sex_cats integer value of total number of sexes possible (in this case it's 2).
#' @param sex_idx integer value identifying the passenger sex.
#' @param survived integer vector of output values, this is binary.
#' @param test_sex_idx integer value identifying the passenger sex from the test dataset.
#' @param test_age Numeric vector of age values from test dataset.
#' @param test_pclass integer value of the test dataset for passenger class status.
#' @param ... Arguments passed to `rstan::sampling` (e.g. iter, chains).
#' 
#' @section Function Descriptions:
#' \describe{
#'   \item{`age_gender_stan()`}{
#'    Age gender Bayesian logistic regression with Stan
#'   }
#'   \item{`age_gender_class_stan()`}{
#'    Age gender and Class Bayesian logistic regression with Stan
#'   }
#' }
NULL

#' @rdname stan-models
#' @export
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

#' @rdname stan-models
#' @export
age_gender_class_stan <- function(age, class_1, class_2, class_3, sex_cats, sex_idx,
                                  survived, test_sex_idx, test_age, test_class_1, test_class_2, test_class_3, ...) {
  standata <- list(N=length(survived),
                   age=age,
                   class_1=class_1,
                   class_2=class_2,
                   class_3=class_3,
                   sex=sex_cats,
                   sex_idx=sex_idx,
                   survived=survived,
                   test_N=length(test_age),
                   test_age=test_age,
                   test_class_1=test_class_1,
                   test_class_2=test_class_2,
                   test_class_3=test_class_3,
                   test_sex_idx=test_sex_idx)
  out <- rstan::sampling(stanmodels$age_gender_class_model, data = standata, ...)
  return(out)
}
