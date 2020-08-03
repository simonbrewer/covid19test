## ----setup, include=FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----message = FALSE------------------------------------------------------------------------
set.seed(1234)
library(dplyr)
library(skimr)
library(caret)
library(gbm) ## For importance scores
library(randomForest)
library(ranger)
library(ggpubr)
library(vip)
library(pdp)


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

# f1 <- test_rate ~ pnursing + wday +
#   pcaseNew_lag + daysSinceC + pdeathNew_lag + daysSinceD + hospRate
# 
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

# modFit <- randomForest(f1, dat = training, do.trace = TRUE)
# plot(modFit)
# varImpPlot(modFit)
# partialPlot(modFit, pred.data = training, x.var = "wday")

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
parGrid = expand.grid(mtry = 2:6, splitrule = "variance", min.node.size = 4:8)
# parGrid = expand.grid(mtry = 6, splitrule = "variance", min.node.size = 4)


ltr_hist <- hist(training$ltest_rate, plot = FALSE,
                 seq(min(training$ltest_rate), max(training$ltest_rate), length.out = 100))
wgt_vec <- 1 / ltr_hist$counts
casewgt <- wgt_vec[cut(training$ltest_rate, include.lowest = TRUE,
                       breaks = ltr_hist$breaks, labels = FALSE)]

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
  num.trees = 250,
  weights = casewgt,
  importance = 'permutation',
  trControl = ctrl
)
modFit


## -------------------------------------------------------------------------------------------
ggplot(modFit)


## -------------------------------------------------------------------------------------------
# plot(varImp(modFit))


## -------------------------------------------------------------------------------------------
pred_test <- predict(modFit, newdata = testing)
# mod_results[mod_id, 2:4] <- postResample(pred = pred_test, obs = testing$test_rate)
print(postResample(pred = exp(pred_test), obs = testing$test_rate))


## -------------------------------------------------------------------------------------------
testing$baseline <- testing$pState_popn
testing$baseline <- ((testing$sTest * testing$pState_popn) / testing$Tot_pop) * 1e3
print(postResample(pred <- testing$baseline, obs = testing$test_rate))


## -------------------------------------------------------------------------------------------
mydf = data.frame(obs = rep(testing$test_rate, 2), 
                  pred = c(testing$baseline, exp(pred_test)),
                  type = rep(c("base", "pred"), each = length(testing$test_rate)))
p1 = ggscatter(mydf, x = "obs", y = "pred", col = "type",
          main = paste0("COVID 19 testing (ranger, raw-scale)")) + 
  geom_abline()
print(p1)

ggsave("./results/COVID19_ranger_cv.pdf", p1)
