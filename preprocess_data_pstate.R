## ----message = FALSE----------------------------------------------------------
set.seed(12345)
library(dplyr)
library(skimr)
library(caret)
library(lubridate)
library(forecastML)
library(zoo)

## -----------------------------------------------------------------------------
## Read and skim data
dat <- read.csv("./rawdata/countyTable_timeSeries_v3.csv")
skim(dat)


## -----------------------------------------------------------------------------
## Remove anything with missing hospital values
dat <- dat[!is.na(dat$hospitals), ]


## -----------------------------------------------------------------------------
## Remove MA and NV (one county each)
dat <- dat %>%
  filter(!state %in% c("MA", "NV"))


## -----------------------------------------------------------------------------
## Remove the IL zero tests
dat <- dat %>%
  filter(!(state == "IL" & cTest == 0))


## -----------------------------------------------------------------------------
## For several states the test numbers are cumulative
## Calculate 'dTest' as the difference between cTest values\
dat$dTest <- dat$cTest
fips <- unique(dat$FIPS)
nfips <- length(fips)
for (i in 1:nfips) {
  fipsid <- which(dat$FIPS == fips[i])
  tmp <- dat[fipsid, ]
  if (!tmp$state[1] %in% c("NY", "CT", "MI", "WA")) {
    tmpdTest <- c(0, diff(tmp$cTest))
    tmpdTest[tmpdTest < 0] <- 0
    dat$dTest[fipsid] <- tmpdTest
  }
}


## -----------------------------------------------------------------------------
state_test <- dat %>% 
  group_by(state, date) %>%
  summarize(state_tests = sum(dTest),
            state_popn = sum(Tot_pop))
## Some states have zero tests - remove these
# state_test$state_tests[state_test$state_tests == 0] <- 10 ## Kludge for missing tests
state_test <- state_test %>% 
  filter(state_tests > 0)


## -----------------------------------------------------------------------------
## Merge state test numbers back to main data frame
dat <- merge(dat, state_test, by = c("state", "date"))


## -----------------------------------------------------------------------------
## Calculate daily test rate
dat$test_rate <- dat$dTest / dat$state_test

## -----------------------------------------------------------------------------
## Calculate other rate data (per 100,000)
## Population data
dat$pState_popn <- dat$Tot_pop / dat$state_popn
dat$pPop_o_60 <- dat$Pop_o_60 / dat$Tot_pop * 1e5
dat$pPop_m <- dat$Pop_m / dat$Tot_pop * 1e5
dat$pPop_white <- dat$Pop_white / dat$Tot_pop * 1e5
dat$pPop_black <- dat$Pop_black / dat$Tot_pop * 1e5
dat$pPop_AmIndAlNat <- dat$Pop_AmIndAlNat / dat$Tot_pop * 1e5
dat$pPop_asia <- dat$Pop_asia / dat$Tot_pop * 1e5
dat$pPop_NaHaPaIs <- dat$Pop_NaHaPaIs / dat$Tot_pop * 1e5

## Education
dat$pBachelor <- dat$Bachelor / dat$Tot_pop * 1e5

## Infrastructure
dat$pnursing <- dat$nursing / dat$Tot_pop * 1e5
dat$phospitals <- dat$hospitals / dat$Tot_pop * 1e5
dat$puniversities <- dat$universities / dat$Tot_pop * 1e5

## Case and death rates
dat$pcaseNew <- dat$caseNew / dat$Tot_pop * 1e5
dat$pdeathNew <- dat$deathNew / dat$Tot_pop * 1e5

dat$pnursing <- dat$nursing / dat$Tot_pop * 1e5
dat$phospitals <- dat$hospitals / dat$Tot_pop * 1e5
dat$puniversities <- dat$universities / dat$Tot_pop * 1e5


## -----------------------------------------------------------------------------
## Log transforms
dat$lpState_popn <- log(dat$pState_popn)
dat$lpPop_o_60 <- log(dat$pPop_o_60)
dat$lpPop_m <- log(dat$pPop_m)
dat$lpPop_white <- log(dat$pPop_white)
dat$lpPop_black <- log(dat$pPop_black + 1e-6)
dat$lpPop_AmIndAlNat <- log(dat$pPop_AmIndAlNat + 1e-6)
dat$lpPop_asia <- log(dat$pPop_asia + 1e-6)
dat$lpPop_NaHaPaIs <- log(dat$pPop_NaHaPaIs + 1e-6)
dat$lIncome <- log(dat$Income)
dat$lpBachelor <- log(dat$pBachelor)


## -----------------------------------------------------------------------------
dat$ddate <- ymd(dat$date)

## -----------------------------------------------------------------------------
## Calculate lagged means (previous 7 days) for cases and deaths
dat$pcaseNew_lag <- rep(NA, nrow(dat))
dat$pdeathNew_lag <- rep(NA, nrow(dat))
mywin = 7 ## previous time steps
for (i in 1:nfips) {
  fipsid <- which(dat$FIPS == fips[i])
  tmp <- dat[fipsid, ]
  
  for (j in 1:nrow(tmp)) {
    ddiff <- tmp$ddate[j] - tmp$ddate
    ddiff_id <- which(ddiff > 0 & ddiff <= mywin)
    # print(ddiff_id)
    if (length(ddiff_id) > 0) {
      tmp$pcaseNew_lag[j] <- mean(tmp$pcaseNew[ddiff_id], na.rm = TRUE)
      tmp$pdeathNew_lag[j] <- mean(tmp$pdeathNew[ddiff_id], na.rm = TRUE)
    }
    
  }
  dat$pcaseNew_lag[fipsid] <- tmp$pcaseNew_lag
  dat$pdeathNew_lag[fipsid] <- tmp$pdeathNew_lag
}


# dat <- fill_gaps(dat, date_col = date_id, frequency = "1 day", groups = "FIPS")

# dat2 <- dat %>%
#   select(ddate, pcaseNew, pdeathNew, hospRate) 
# %>% 
#   create_lagged_df(type = "train", lookback = 7, 
#                    horizons = 1, groups = "FIPS",
#                    dates = dat$ddate)

save(dat, file = "covid19.RData")

