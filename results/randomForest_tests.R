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

# dat <- dat[sample(nrow(dat), 5000), ]
dat$ltest_rate <- log(dat$test_rate+1e-1)

dat1 <- dat %>%
  filter(state %in% c("CT", "MI", "NY", "WA"))

dat2 <- dat %>%
  filter(!state %in% c("CT", "MI", "NY", "WA"))


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
  y = dat1$ltest_rate,
  ## the outcome data are needed
  p = .75,
  ## The percentage of data in the
  ## training set
  list = FALSE
)

training <- dat1[ inTrain,]
testing  <- dat1[-inTrain,]

modFit <- randomForest(f1, dat = training, 
                       ntree = 250,
                       do.trace = TRUE,
                       corr.bias = TRUE)
plot(modFit)
varImpPlot(modFit)
pdf("./results/COVID19_rf_vip.pdf")
vip(modFit)
dev.off()
pdf("./results/COVID19_rf_pdp_wday.pdf")
autoplot(partial(modFit, pred.var = "wday"), 
            main = "Partial dependency: weekday")
dev.off()
# partialPlot(partialmodFit, pred.data = training, x.var = "wday", 
#             main = "Partial dependency: weekday")

pred_test <- predict(modFit, testing)
print(postResample(pred = exp(pred_test), obs = testing$test_rate))
testing$baseline <- ((testing$sTest * testing$pState_popn) / testing$Tot_pop) * 1e3
print(postResample(pred = testing$baseline, obs = testing$test_rate))

## -------------------------------------------------------------------------------------------
mydf = data.frame(obs = rep(testing$test_rate, 2), 
                  pred = c(testing$baseline, exp(pred_test)),
                  type = rep(c("base", "pred"), each = length(testing$test_rate)))
p1 = ggscatter(mydf, x = "obs", y = "pred", col = "type",
               main = paste0("COVID 19 testing (random forest, raw-scale)")) + 
  geom_abline()
print(p1)


## -------------------------------------------------------------------------------------------
training <- dat1[ inTrain,]
testing  <- dat1[-inTrain,]

ltr_hist <- hist(training$ltest_rate, plot = FALSE,
                 seq(min(training$ltest_rate), max(training$ltest_rate), length.out = 10))
wgt_vec <- 1 / ltr_hist$counts
rfcasewgt <- wgt_vec[cut(training$ltest_rate, include.lowest = TRUE,
                          breaks = ltr_hist$breaks, labels = FALSE)]

modFit <- ranger(f1, data = training, num.trees = 250, case.weights = rfcasewgt)

# plot(modFit)
# varImpPlot(modFit)
# partialPlot(modFit, pred.data = training, x.var = "wday")

pred_test <- predict(modFit, testing)$predictions
print(postResample(pred = exp(pred_test), obs = testing$test_rate))
testing$baseline <- ((testing$state_tests * testing$pState_popn) / testing$Tot_pop) * 1e3
print(postResample(pred = testing$baseline, obs = testing$test_rate))

## -------------------------------------------------------------------------------------------
mydf = data.frame(obs = rep(testing$test_rate, 2), 
                  pred = c(testing$baseline, exp(pred_test)),
                  type = rep(c("base", "pred"), each = length(testing$test_rate)))
p1 = ggscatter(mydf, x = "obs", y = "pred", col = "type",
               main = paste0("COVID 19 testing (random forest, raw-scale)")) + 
  geom_abline()
print(p1)

stop()
## -------------------------------------------------------------------------------------------
## Full model
# modFit <- randomForest(f1, dat = dat1, 
#                        ntree = 250,
#                        do.trace = TRUE,
#                        corr.bias = TRUE)
ltr_hist <- hist(dat1$ltest_rate, plot = FALSE,
                 seq(min(dat1$ltest_rate), max(dat1$ltest_rate), length.out = 100))
wgt_vec <- 1 / ltr_hist$counts
rfcasewgt <- wgt_vec[cut(dat1$ltest_rate, include.lowest = TRUE,
                         breaks = ltr_hist$breaks, labels = FALSE)]

modFit <- ranger(f1, data = dat1, num.trees = 250, 
                 case.weights = rfcasewgt, 
                 importance = "permutation")
# modFit <- ranger(f1, data = dat1, num.trees = 250)

## -------------------------------------------------------------------------------------------
states = unique(sort(dat2$state))
nstates = length(states)

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

for (i in 1:nstates) {
  print(states[i])
  state_id = which(dat2$state == states[i])
  # training = dat[-state_id,]
  testing =  dat2[ state_id,]
  
  ## Get baseline
  # testing$baseline = testing$pState_popn
  testing$baseline <- ((testing$sTest * testing$pState_popn) / testing$Tot_pop) * 1e3
  
  ## Plot tuning results
  # ggplot(modFit)
  pred_test <- predict(modFit, data = testing)$predictions
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

