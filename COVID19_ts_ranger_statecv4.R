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

dat <- dat %>%
  filter(!is.na(pcaseNew_lag))

dat$ltest_rate <- log(dat$test_rate+1e-5)

dat1 <- dat %>%
  filter(state %in% c("CT", "MI", "NY", "WA"))

dat2 <- dat %>%
  filter(!state %in% c("CT", "MI", "NY", "WA"))

## -------------------------------------------------------------------------------------------
f1 <- ltest_rate ~ lpState_popn + lpPop_o_60 + lpPop_m + lpPop_white + 
  lpPop_black + lpPop_AmIndAlNat + lpPop_asia + lpPop_NaHaPaIs +
  lIncome + lpBachelor + phospitals + pnursing + puniversities +
  pcaseNew_lag + daysSinceC + pdeathNew_lag + daysSinceD + hospRate + wday #+ sTest + Tot_pop

# f1 <- test_rate ~ pnursing +
#   pcaseNew + daysSinceC + pdeathNew + daysSinceD + hospRate

## -------------------------------------------------------------------------------------------
states = unique(sort(dat2$state))
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
parGrid = expand.grid(mtry = 5, splitrule = "variance", min.node.size = 4)

ltr_hist <- hist(dat1$ltest_rate, plot = FALSE,
                 seq(min(dat1$ltest_rate), max(dat1$ltest_rate), length.out = 100))
wgt_vec <- 1 / ltr_hist$counts
casewgt <- wgt_vec[cut(dat1$ltest_rate, include.lowest = TRUE,
                       breaks = ltr_hist$breaks, labels = FALSE)]

modFit <- train(
  f1,
  data = dat1,
  method = "ranger",
  ## Center and scale the predictors for the training
  ## set and all future samples.
  preProc = c("center", "scale"),
  ## increase parameter set
  tuneGrid = parGrid,
  weights = casewgt,
  ## added:
  # importance = 'permutation',
  trControl = ctrl)



## -------------------------------------------------------------------------------------------
for (i in 1:nstates) {
  print(states[i])
  state_id = which(dat$state == states[i])
  # training = dat[-state_id,]
  testing =  dat[ state_id,]
  
  ## Get baseline
  # testing$baseline = testing$pState_popn
  testing$baseline <- ((testing$sTest * testing$pState_popn) / testing$Tot_pop) * 1e3
  
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

