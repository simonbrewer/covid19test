## Tuning lme4 models (not really tuning)
## ----setup, include=FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----message = FALSE------------------------------------------------------------------------
set.seed(1234)
library(dplyr)
library(skimr)
library(caret)
library(lme4)
library(lmerTest)
library(ggpubr)


load("./covid19.RData")
dat <- dat %>%
  filter(!is.na(pcaseNew_lag))

# dat <- dat %>%
#   filter(state %in% c("CT", "MI", "NY", "WA"))

# dat <- dat[sample(nrow(dat), 5000), ]
dat$ltest_rate <- log(dat$test_rate+1e-5)
## -------------------------------------------------------------------------------------------
f1 <- ltest_rate ~ lpState_popn + 
  lIncome + phospitals +
  pcaseNew_lag + daysSinceC + wday +
  (1 | FIPS) # + sTest

# f1 <- ltest_rate ~ pState_popn * daysSinceC + pcaseNew_lag +
#   (1 | FIPS)
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

modFit <- lmer(f1, training)

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
          main = paste0("COVID 19 testing (lme4, raw-scale)")) + 
  geom_abline()
print(p1)

ggsave("./results/COVID19_lme_cv.pdf", p1)
