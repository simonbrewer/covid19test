## ----setup, include=FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----message = FALSE------------------------------------------------------------------------
set.seed(12345)
library(dplyr)
library(skimr)
library(caret)
library(gbm) ## For importance scores
library(ranger)
library(ggpubr)
library(vip)
library(pdp)


load("covid19.RData")

## -------------------------------------------------------------------------------------------
f1 <- test_rate ~ pState_popn + lpPop_o_60 + lpPop_m + lpPop_white + 
  # lpPop_black + lpPop_AmIndAlNat + lpPop_asia + lpPop_NaHaPaIs + 
  lIncome + lpBachelor + phospitals + pnursing + puniversities + 
  pcaseNew + daysSinceC + pdeathNew + daysSinceD # + hospRate


## -------------------------------------------------------------------------------------------
inTrain <- createDataPartition(
  y = dat$test_rate,
  ## the outcome data are needed
  p = .75,
  ## The percentage of data in the
  ## training set
  list = FALSE
)

training <- dat[ inTrain,]
testing  <- dat[-inTrain,]


## -------------------------------------------------------------------------------------------
# ctrl <- trainControl(method = "repeatedcv", 
#                      number = 5,
#                      repeats = 3,
#                      verboseIter = TRUE)
ctrl <- trainControl(method = "cv", 
                     number = 3,
                     verboseIter = TRUE)
# ctrl <- trainControl(method = "none", verboseIter = TRUE)


## -------------------------------------------------------------------------------------------
parGrid = expand.grid(mtry = 6:10, splitrule = "variance", min.node.size = 2:6)
# parGrid = expand.grid(mtry = 6, splitrule = "variance", min.node.size = 4)


## ----results='hide', message=FALSE----------------------------------------------------------
modFit <- train(
  f1,
  data = training,
  method = "ranger",
  ## Center and scale the predictors for the training
  ## set and all future samples.
  preProc = c("center", "scale"),
  ## increase parameter set
  tuneGrid = parGrid,
  ## added:
  # importance = 'permutation',
  trControl = ctrl
)
modFit


## -------------------------------------------------------------------------------------------
ggplot(modFit)


## -------------------------------------------------------------------------------------------
#plot(varImp(modFit))


## -------------------------------------------------------------------------------------------
pred_test <- predict(modFit, newdata = testing)
# mod_results[mod_id, 2:4] <- postResample(pred = pred_test, obs = testing$test_rate)
postResample(pred = pred_test, obs = testing$test_rate)


## -------------------------------------------------------------------------------------------
testing$baseline = testing$pState_popn
postResample(pred = testing$baseline, obs = testing$test_rate)


## -------------------------------------------------------------------------------------------
mydf = data.frame(obs = rep(testing$test_rate, 2), 
                  pred = c(testing$baseline, pred_test),
                  type = rep(c("base", "pred"), each = length(testing$test_rate)))
ggscatter(mydf, x = "obs", y = "pred", col = "type",
          main = paste0("COVID 19 testing (ranger, raw-scale)")) + 
  geom_abline()

