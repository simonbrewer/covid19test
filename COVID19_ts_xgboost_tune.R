## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----message = FALSE----------------------------------------------------------
set.seed(12345)
library(dplyr)
library(skimr)
library(caret)
library(xgboost) ## For importance scores
library(ggpubr)
library(vip)
library(pdp)


## -----------------------------------------------------------------------------
load("./covid19.RData")


## -----------------------------------------------------------------------------
f1 <- test_rate ~ lpState_popn + lpPop_o_60 + lpPop_m + lpPop_white + 
#  lpPop_black + lpPop_AmIndAlNat + lpPop_asia + lpPop_NaHaPaIs + 
  lIncome + lpBachelor + phospitals + pnursing + puniversities + 
  pcaseNew + daysSinceC + pdeathNew + daysSinceD + hospRate


## -----------------------------------------------------------------------------
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


## -----------------------------------------------------------------------------
ctrl <- trainControl(method = "repeatedcv", 
                     number = 5,
                     repeats = 3,
                     verboseIter = TRUE)
ctrl <- trainControl(method = "cv", 
                     number = 3,
                     allowParallel = TRUE,
                     verboseIter = TRUE)

# ## -----------------------------------------------------------------------------
# ## Round 1: general tuning
# parGrid <- expand.grid(
#   nrounds = seq(200, 1000, by = 50), 
#   eta = c(1e-4, 1e-3, 1e-2, 1e-1), 
#   max_depth = c(2, 3, 4, 5, 6), 
#   gamma = 0, 
#   colsample_bytree = 1, 
#   min_child_weight = 1,
#   subsample = 1)
# 
# 
# ## ----results='hide', message=FALSE--------------------------------------------
# modFit <- train(
#   f1,
#   data = training,
#   method = "xgbTree",
#   ## Center and scale the predictors for the training
#   ## set and all future samples.
#   preProc = c("center", "scale"),
#   ## increase parameter set
#   tuneGrid = parGrid,
#   ## added:
#   trControl = ctrl
# )
# modFit
# 
# 
# ## -----------------------------------------------------------------------------
# ggplot(modFit)
# 
# 
# ## -----------------------------------------------------------------------------
# ## Round 2: child weight
# parGrid <- expand.grid(
#   nrounds = seq(50, 1000, by = 50), 
#   eta = 0.1, 
#   max_depth = 3, 
#   gamma = 0, 
#   colsample_bytree = 1, 
#   min_child_weight = c(1, 2, 3, 4),
#   subsample = 1)
# 
# 
# ## ----results='hide', message=FALSE--------------------------------------------
# modFit <- train(
#   f1,
#   data = training,
#   method = "xgbTree",
#   ## Center and scale the predictors for the training
#   ## set and all future samples.
#   preProc = c("center", "scale"),
#   ## increase parameter set
#   tuneGrid = parGrid,
#   ## added:
#   trControl = ctrl
# )
# modFit
# 
# 
# ## -----------------------------------------------------------------------------
# ggplot(modFit)
# 
# 
# ## -----------------------------------------------------------------------------
# ## Round 3: row and column sampling
# parGrid <- expand.grid(
#   nrounds = seq(50, 1000, by = 50), 
#   eta = 0.1, 
#   max_depth = 3, 
#   gamma = 0, 
#   colsample_bytree = c(0.4, 0.6, 0.8, 1.0), 
#   min_child_weight = 1,
#   subsample = c(0.5, 0.75, 1.0)
#   )
# 
# 
# ## ----results='hide', message=FALSE--------------------------------------------
# modFit <- train(
#   f1,
#   data = training,
#   method = "xgbTree",
#   ## Center and scale the predictors for the training
#   ## set and all future samples.
#   preProc = c("center", "scale"),
#   ## increase parameter set
#   tuneGrid = parGrid,
#   ## added:
#   trControl = ctrl
# )
# modFit
# 
# 
# ## -----------------------------------------------------------------------------
# ggplot(modFit)
# 
# 
# ## -----------------------------------------------------------------------------
# ## Round 4: gamma
# parGrid <- expand.grid(
#   nrounds = seq(50, 1000, by = 50), 
#   eta = 0.1, 
#   max_depth = 3, 
#   gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0), 
#   colsample_bytree = 1.0, 
#   min_child_weight = 3,
#   subsample = 1.0
#   )
# 
# 
# ## ----results='hide', message=FALSE--------------------------------------------
# modFit <- train(
#   f1,
#   data = training,
#   method = "xgbTree",
#   ## Center and scale the predictors for the training
#   ## set and all future samples.
#   preProc = c("center", "scale"),
#   ## increase parameter set
#   tuneGrid = parGrid,
#   ## added:
#   trControl = ctrl
# )
# modFit
# 
# 
# ## -----------------------------------------------------------------------------
# ggplot(modFit)


## -----------------------------------------------------------------------------
## Round 5: learning rate
parGrid <- expand.grid(
  nrounds = seq(100, 10000, by = 100), 
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1), 
  max_depth = 3, 
  gamma = 0, 
  colsample_bytree = 1.0, 
  min_child_weight = 3,
  subsample = 1.0
  )


## ----results='hide', message=FALSE--------------------------------------------
modFit <- train(
  f1,
  data = training,
  method = "xgbTree",
  ## Center and scale the predictors for the training
  ## set and all future samples.
  preProc = c("center", "scale"),
  ## increase parameter set
  tuneGrid = parGrid,
  ## added:
  trControl = ctrl
)
modFit


## -----------------------------------------------------------------------------
ggplot(modFit)


## -----------------------------------------------------------------------------
plot(varImp(modFit))


## -----------------------------------------------------------------------------
pred_test <- predict(modFit, newdata = testing)
# mod_results[mod_id, 2:4] <- postResample(pred = pred_test, obs = testing$test_rate)
postResample(pred = pred_test, obs = testing$test_rate)

## -------------------------------------------------------------------------------------------
testing$baseline = testing$pState_popn
postResample(pred = testing$baseline, obs = testing$test_rate)


## -----------------------------------------------------------------------------
mydf = data.frame(obs = testing$test_rate, pred = pred_test)
p1 = ggscatter(mydf, x = "obs", y = "pred", 
          main = paste0("COVID 19 testing (gbm, raw-scale)")) + 
  geom_abline()
ggsave("COVID19_xgboost_simp_cv.pdf", p1)
