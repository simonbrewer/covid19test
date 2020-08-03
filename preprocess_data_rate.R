## New preprocessing script that just uses true test rates
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
dat <- read.csv("./rawdata/countyTable_timeSeries_v4.csv")
skim(dat)

state_popn <- read.csv("./rawdata/state_popn_2019.csv")

## -----------------------------------------------------------------------------
## Remove anything with missing hospital values
dat <- dat[!is.na(dat$hospitals), ]


## -----------------------------------------------------------------------------
## Remove MA and NV (one county each)
# dat <- dat %>%
#   filter(!state %in% c("MA", "NV"))


## -----------------------------------------------------------------------------
## Remove the IL zero tests
# dat <- dat %>%
#   filter(!(state == "IL" & cTest == 0))

## -----------------------------------------------------------------------------
## Drop the 15/16 May from Florida
# dat <- dat %>%
#   filter(!(state == "FL" & date %in% c("2020-05-15", "2020-05-16")))


# ## -----------------------------------------------------------------------------
# ## For several states the test numbers are cumulative
# ## Calculate 'dTest' as the difference between cTest values\
# dat$dTest <- dat$cTest
# fips <- unique(dat$FIPS)
# nfips <- length(fips)
# for (i in 1:nfips) {
#   fipsid <- which(dat$FIPS == fips[i])
#   tmp <- dat[fipsid, ]
#   if (!tmp$state[1] %in% c("NY", "CT", "MI", "WA")) {
#     tmpdTest <- c(0, diff(tmp$cTest))
#     tmpdTest[tmpdTest < 0] <- 0
#     dat$dTest[fipsid] <- tmpdTest
#   }
# }


## -----------------------------------------------------------------------------
state_test <- dat %>% 
  group_by(state, date) %>%
  summarize(state_tests = sum(sTest),
            state_popn = sum(Tot_pop))
## Some states have zero tests - remove these
# state_test$state_tests[state_test$state_tests == 0] <- 10 ## Kludge for missing tests
state_test <- state_test %>% 
  filter(state_tests > 0)

## -----------------------------------------------------------------------------
## Merge census based state population values
state_test <- merge(state_test, state_popn, by.x = "state", by.y = "Abbr")
state_test <- state_test %>%
  select(-State, -sFIPS)


## -----------------------------------------------------------------------------
## Merge state test numbers back to main data frame
dat <- merge(dat, state_test, by = c("state", "date"))

## Filter by state tests = 0
# dat <- dat %>%
#   filter(sTest > 0)

## -----------------------------------------------------------------------------
## Calculate daily test rate
dat$test_rate <- (dat$cTest / dat$Tot_pop) * 1e3

## -----------------------------------------------------------------------------
## Calculate other rate data (per 100,000)
## Population data
dat$pState_popn <- dat$Tot_pop / dat$Popn
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
# dat$ddate <- ymd(dat$date)
dat$ddate <- mdy(dat$date)
dat$wday <- wday(dat$ddate, week_start = 1, label = TRUE)

## -----------------------------------------------------------------------------
## Calculate lagged means (previous 7 days) for cases and deaths
dat$pcaseNew_lag <- rep(NA, nrow(dat))
dat$pdeathNew_lag <- rep(NA, nrow(dat))
mywin = 7 ## previous time steps
fips <- unique(dat$FIPS)
nfips <- length(fips)
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

