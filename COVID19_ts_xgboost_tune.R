## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----message = FALSE----------------------------------------------------------
set.seed(1234)
library(dplyr)
library(skimr)
library(caret)
library(xgboost) ## For importance scores
library(ggpubr)
library(vip)
library(pdp)


## -----------------------------------------------------------------------------
load("./covid19.RData")

dat <- dat %>%
  filter(!is.na(pcaseNew_lag))

# dat <- dat %>%
#   filter(state %in% c("CT", "MI", "NY", "WA"))

# dat <- dat[sample(nrow(dat), 5000), ]
dat$ltest_rate <- log(dat$test_rate+1e-5)
## -------------------------------------------------------------------------------------------
f1 <- ltest_rate ~ lpState_popn + lpPop_o_60 + lpPop_m + lpPop_white + 
  lpPop_black + lpPop_AmIndAlNat + lpPop_asia + lpPop_NaHaPaIs +
  lIncome + lpBachelor + phospitals + pnursing + puniversities +
  pcaseNew_lag + daysSinceC + pdeathNew_lag + daysSinceD + hospRate + wday # + sTest

# f1 <- test_rate ~ pnursing +
#   pcaseNew_lag + daysSinceC + pdeathNew_lag + daysSinceD + hospRate

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
## Case weights
ltr_hist <- hist(training$ltest_rate, plot = FALSE,
                 seq(min(training$ltest_rate), max(training$ltest_rate), length.out = 100))
wgt_vec <- 1 / ltr_hist$counts
casewgt <- wgt_vec[cut(training$ltest_rate, include.lowest = TRUE,
                       breaks = ltr_hist$breaks, labels = FALSE)]


## -----------------------------------------------------------------------------
ctrl <- trainControl(method = "repeatedcv", 
                     number = 5,
                     repeats = 3,
                     verboseIter = TRUE)
ctrl <- trainControl(method = "cv", 
                     number = 3,
                     allowParallel = TRUE,
                     verboseIter = TRUE)

## -----------------------------------------------------------------------------
## Round 1: general tuning
parGrid <- expand.grid(
  nrounds = seq(200, 1000, by = 50),
  eta = c(1e-4, 1e-3, 1e-2, 1e-1),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1)


## ----results='hide', message=FALSE--------------------------------------------
modFit1 <- train(
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
modFit1


## -----------------------------------------------------------------------------
ggplot(modFit1)


## -----------------------------------------------------------------------------
## Round 2: child weight
parGrid <- expand.grid(
  nrounds = seq(50, 1000, by = 50),
  eta = modFit1$bestTune$eta,
  max_depth = modFit1$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3, 4),
  subsample = 1)


## ----results='hide', message=FALSE--------------------------------------------
modFit2 <- train(
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
modFit2


## -----------------------------------------------------------------------------
ggplot(modFit2)


## -----------------------------------------------------------------------------
## Round 3: row and column sampling
parGrid <- expand.grid(
  nrounds = seq(50, 1000, by = 50),
  eta = modFit1$bestTune$eta,
  max_depth = modFit1$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = modFit2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
  )


## ----results='hide', message=FALSE--------------------------------------------
modFit3 <- train(
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
modFit3


## -----------------------------------------------------------------------------
ggplot(modFit3)


## -----------------------------------------------------------------------------
## Round 4: gamma
parGrid <- expand.grid(
  nrounds = seq(50, 1000, by = 50),
  eta = modFit1$bestTune$eta,
  max_depth = modFit1$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = modFit3$bestTune$colsample_bytree,
  min_child_weight = modFit2$bestTune$min_child_weight,
  subsample = modFit3$bestTune$subsample
  )


## ----results='hide', message=FALSE--------------------------------------------
modFit4 <- train(
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
modFit4


## -----------------------------------------------------------------------------
ggplot(modFit4)


## -----------------------------------------------------------------------------
## Round 5: learning rate
parGrid <- expand.grid(
  nrounds = seq(100, 2000, by = 100), 
  # eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  eta = 0.025,
  max_depth = modFit1$bestTune$max_depth,
  gamma = modFit4$bestTune$gamma, 
  colsample_bytree = modFit3$bestTune$colsample_bytree,
  min_child_weight = modFit2$bestTune$min_child_weight,
  subsample = modFit3$bestTune$subsample
)


## ----results='hide', message=FALSE--------------------------------------------
modFit5 <- train(
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
modFit5


## -----------------------------------------------------------------------------
ggplot(modFit5)


## -----------------------------------------------------------------------------
plot(varImp(modFit5))


## -----------------------------------------------------------------------------
pred_test <- predict(modFit5, newdata = testing)
# mod_results[mod_id, 2:4] <- postResample(pred = pred_test, obs = testing$test_rate)
postResample(pred = exp(pred_test), obs = testing$test_rate)

## -------------------------------------------------------------------------------------------
testing$baseline = testing$pState_popn
testing$baseline <- ((testing$sTest * testing$pState_popn) / testing$Tot_pop) * 1e3
postResample(pred = testing$baseline, obs = testing$test_rate)


## -----------------------------------------------------------------------------
mydf = data.frame(obs = rep(testing$test_rate, 2), 
                  pred = c(testing$baseline, exp(pred_test)),
                  type = rep(c("base", "pred"), each = length(testing$test_rate)))
p1 = ggscatter(mydf, x = "obs", y = "pred", col = "type",
               main = paste0("COVID 19 testing (xgboost, raw-scale)")) + 
  geom_abline()
print(p1)

ggsave("./results/COVID19_xgboost_cv.pdf", p1)

