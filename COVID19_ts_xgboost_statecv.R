## ----setup, include=FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----message = FALSE------------------------------------------------------------------------
set.seed(12345)
library(dplyr)
library(skimr)
library(caret)
library(gbm) ## For importance scores
library(ggpubr)
library(vip)
library(pdp)


load("./covid19.RData")

## -------------------------------------------------------------------------------------------
f1 <- test_rate ~ lpState_popn + lpPop_o_60 + lpPop_m + lpPop_white + 
  # lpPop_black + lpPop_AmIndAlNat + lpPop_asia + lpPop_NaHaPaIs + 
  lIncome + lpBachelor + phospitals + pnursing + puniversities +
  pcaseNew + daysSinceC + pdeathNew + daysSinceD + hospRate


## -------------------------------------------------------------------------------------------
states = unique(sort(dat$state))
nstates = length(states)


## -------------------------------------------------------------------------------------------
ctrl <- trainControl(method = "repeatedcv", 
                     number = 5,
                     repeats = 3,
                     verboseIter = TRUE)
ctrl <- trainControl(method = "none", verboseIter = TRUE)


## -------------------------------------------------------------------------------------------
mod_results <- data.frame(states,
                          RMSE = rep(NA, nstates),
                          Rsquared = rep(NA, nstates),
                          MAE = rep(NA, nstates)
)


## -------------------------------------------------------------------------------------------
base_results <- data.frame(states,
                          RMSE = rep(NA, nstates),
                          Rsquared = rep(NA, nstates),
                          MAE = rep(NA, nstates)
)


## -------------------------------------------------------------------------------------------
parGrid <- expand.grid(
  nrounds = 800, 
  eta = 0.05, 
  max_depth = 3, 
  gamma = 0, 
  colsample_bytree = 1.0, 
  min_child_weight = 3,
  subsample = 1.0
)

## -------------------------------------------------------------------------------------------
for (i in 1:nstates) {
  print(states[i])
  state_id = which(dat$state == states[i])
  training = dat[-state_id,]
  testing =  dat[ state_id,]
  
  ## Get baseline
  testing$baseline = testing$pState_popn
  
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
    # importance = 'permutation',
    trControl = ctrl)
  
  ## Plot tuning results
  # ggplot(modFit)
  pred_test <- predict(modFit, newdata = testing)
  results <- postResample(pred = pred_test, obs = testing$test_rate)
  mod_results$RMSE[i] <- results[1]
  mod_results$Rsquared[i] <- results[2]
  mod_results$MAE[i] <- results[3]
  
  results <- postResample(pred = testing$baseline, obs = testing$test_rate)
  base_results$RMSE[i] <- results[1]
  base_results$Rsquared[i] <- results[2]
  base_results$MAE[i] <- results[3]
  
  mydf = data.frame(obs = rep(testing$test_rate, 2), 
                    pred = c(testing$baseline, pred_test),
                    type = rep(c("base", "pred"), each = length(testing$baseline)))
  p1 = ggscatter(mydf, x = "obs", y = "pred", col = "type", 
            main = paste0("COVID 19 testing (",states[i],", raw-scale)")) + 
    geom_abline()
  ggsave(file = paste0("./statecv/xgboost/plots/covid_",states[i],"_ranger.png"), 
         p1, device = "png")
}


## -------------------------------------------------------------------------------------------
knitr::kable(mod_results, digits = 4)
knitr::kable(base_results, digits = 4)

print(paste(mean(base_results$MAE), mean(mod_results$MAE)))

save(mod_results, base_results, file = "./statecv/xgboost/results.RData")

