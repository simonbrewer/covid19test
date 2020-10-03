## ----setup, include=FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----message = FALSE------------------------------------------------------------------------
set.seed(1234)
library(dplyr)
library(skimr)
library(caret)
library(gbm) ## For importance scores
library(ggpubr)
library(vip)
library(pdp)


load("./covid19.RData")

dat <- dat %>%
  filter(!is.na(pcaseNew_lag))

dat$ltest_rate <- log(dat$test_rate+1e-5)

# dat <- dat %>%
#   filter(state %in% c("CT", "MI", "NY", "WA"))

## -------------------------------------------------------------------------------------------
f1 <- ltest_rate ~ lpState_popn + lpPop_o_60 + lpPop_m + lpPop_white + 
  lpPop_black + lpPop_AmIndAlNat + lpPop_asia + lpPop_NaHaPaIs +
  lIncome + lpBachelor + phospitals + pnursing + puniversities +
  pcaseNew_lag + daysSinceC + pdeathNew_lag + daysSinceD + hospRate + wday # + sTest

# f1 <- test_rate ~ pnursing +
#   pcaseNew + daysSinceC + pdeathNew + daysSinceD + hospRate

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
parGrid = expand.grid(mtry = 6, splitrule = "variance", min.node.size = 4)

## -------------------------------------------------------------------------------------------
for (i in 1:nstates) {
  print(states[i])
  state_id = which(dat$state == states[i])
  training = dat[-state_id,]
  testing =  dat[ state_id,]
  
  ltr_hist <- hist(training$ltest_rate, plot = FALSE,
                   seq(min(training$ltest_rate), max(training$ltest_rate), length.out = 100))
  wgt_vec <- 1 / ltr_hist$counts
  casewgt <- wgt_vec[cut(training$ltest_rate, include.lowest = TRUE,
                         breaks = ltr_hist$breaks, labels = FALSE)]
  
  ## Get baseline
  # testing$baseline = testing$pState_popn
  testing$baseline <- ((testing$sTest * testing$pState_popn) / testing$Tot_pop) * 1e3
  
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
    # importance = 'permutation',
    trControl = ctrl)
  
  ## Plot tuning results
  # ggplot(modFit)
  pred_test <- predict(modFit, newdata = testing)
  results <- postResample(pred = exp(pred_test), obs = testing$test_rate)
  # results <- postResample(pred = exp(pred_test), obs = testing$test_rate)
  mod_results$RMSE[i] <- results[1]
  mod_results$Rsquared[i] <- results[2]
  mod_results$MAE[i] <- results[3]
  
  results <- postResample(pred = testing$baseline, obs = testing$test_rate)
  base_results$RMSE[i] <- results[1]
  base_results$Rsquared[i] <- results[2]
  base_results$MAE[i] <- results[3]
  
  mydf = data.frame(obs = rep(testing$test_rate, 2), 
                    pred = c(testing$baseline, exp(pred_test)),
                    type = rep(c("base", "pred"), each = length(testing$baseline)))
  p1 = ggscatter(mydf, x = "obs", y = "pred", col = "type", 
            main = paste0("COVID 19 testing (",states[i],", raw-scale)")) + 
    geom_abline()
  ggsave(file = paste0("./statecv/ranger/covid_",states[i],"_ranger.png"), 
         p1, device = "png")
}


## -------------------------------------------------------------------------------------------
knitr::kable(mod_results, digits = 4)
knitr::kable(base_results, digits = 4)

print(paste(mean(base_results$MAE), mean(mod_results$MAE)))

save(mod_results, base_results, file = "./statecv/ranger/results.RData")

