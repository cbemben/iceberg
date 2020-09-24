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
  data$Age <- ave(data$Age, data$Embarked, data$Pclass,
                  FUN = function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
  data
}
