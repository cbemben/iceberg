#' Impute Passenger Age
#'
#' Fill in null age values with some estimate of the passengers
#' age.
#'
#' @param data a dataframe of passenger details that include age.
#'
#' @export
#'
impute_passenger_age <- function(data){
  data$Age[is.na(data$Age)] <- with(data,
                                      ave(Age, Embarked, Pclass,
                                          FUN = function(x)
                                            median(x, na.rm = TRUE)))
  data
}
