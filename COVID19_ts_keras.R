## ----message = FALSE------------------------------------------------------------------------
set.seed(1234)
library(reticulate)
library(dplyr)
library(skimr)
library(caret)
library(keras)
library(tibble)
library(recipes)
library(readr)
library(ggpubr)


batch_size = 16
total_epochs = 20

### Data processing

load("./covid19.RData")
dat <- dat %>%
  filter(!is.na(pcaseNew_lag))

dat <- dat %>%
  filter(state %in% c("CT", "MI", "NY", "WA"))

# dat <- dat[sample(nrow(dat), 5000), ]
dat$ltest_rate <- log(dat$test_rate+1e-1)
# dat <- dat %>% 
#   filter(test_rate > 0)
# dat$ltest_rate <- log(dat$test_rate)

## -------------------------------------------------------------------------------------------
inTrain <- createDataPartition(
  y = dat$ltest_rate,
  ## the outcome data are needed
  p = .75,
  ## The percentage of data in the
  ## training set
  list = FALSE
)

training <- dat[ inTrain,]
testing  <- dat[-inTrain,]

## -------------------------------------------------------------------------------------------
f1 <- test_rate ~ lpState_popn + lpPop_o_60 + lpPop_m + lpPop_white + 
  lpPop_black + lpPop_AmIndAlNat + lpPop_asia + lpPop_NaHaPaIs +
  lIncome + lpBachelor + phospitals + pnursing + puniversities +
  pcaseNew_lag + daysSinceC + pdeathNew_lag + daysSinceD + hospRate + wday + sTest

model_recipe <- recipe(f1, 
                       data = dat)

# define the steps we want to apply
model_recipe_steps <- model_recipe %>% 
  # mean impute numeric variables
  # step_meanimpute(all_numeric()) %>%
  # convert the additional ingredients variable to dummy variables
  step_dummy(all_nominal()) %>%
  # rescale all numeric variables except for test rate
  step_range(all_numeric(), min = 0, max = 1, -test_rate) %>%
  # step_range(all_numeric(), min = 0, max = 1) %>%
  # remove predictor variables that are almost the same for every entry
  step_nzv(all_predictors()) 

prepped_recipe <- prep(model_recipe_steps, training = training)
prepped_recipe

X_train <- bake(prepped_recipe, training) 
y_train = X_train$test_rate
X_train <- X_train %>%
  select(-test_rate)
X_train <- data.matrix(X_train)

X_test <- bake(prepped_recipe, testing) 
y_test = X_test$test_rate
X_test <- X_test %>%
  select(-test_rate)
X_test <- data.matrix(X_test)

# dimensions of our input data
dim(X_train)



## -------------------------------------------------------------------------------------------
# initialize our model
model <- keras_model_sequential()

# our input layer
# model %>%
#   layer_dense(units = 256,
#               input_shape = dim(X_train)[[2]],
#               kernel_regularizer = regularizer_l2(0.001), 
#               activation = "relu")
# 
# model %>% layer_dropout(rate = 0.2) 

# model %>%
#   layer_dense(units = 128,
#               input_shape = dim(X_train)[[2]],
#               kernel_regularizer = regularizer_l2(0.001), 
#               activation = "relu")
# 
# model %>% layer_dropout(rate = 0.2) 
# 
# model %>%
#   layer_dense(units = 128,
#               input_shape = dim(X_train)[[2]],
#               kernel_regularizer = regularizer_l2(0.001), 
#               activation = "relu")
# 
# model %>% layer_dropout(rate = 0.2) 
# 
model %>%
  layer_dense(units = 64,
              input_shape = dim(X_train)[[2]],
              kernel_initializer = "normal",
              bias_initializer = initializer_constant(0.1),
              kernel_regularizer = regularizer_l2(0.001), 
              activation = "relu")
model %>% layer_dropout(rate = 0.2) 

# model %>% layer_dense(units = 64,
#                       kernel_regularizer = regularizer_l2(0.001),
#                       activation = "relu")
# model %>% layer_dropout(rate = 0.2)

model %>% layer_dense(units = 32,
                      kernel_initializer = "normal",
                      kernel_regularizer = regularizer_l2(0.001),
                      activation = "relu")
model %>% layer_dropout(rate = 0.2)

model %>%
  layer_dense(units = 1, activation = "exponential") # output

# look at our model architecture
summary(model)

## Comile it
model %>% compile(
  optimizer = optimizer_adamax(lr = 1e-3), 
  loss = "mse", 
  metrics = c("mae")
)

## -------------------------------------------------------------------------------------------
# Actually train our model! This step will take a while
trained_model <- model %>% fit(
  x = X_train, # sequence we're using for prediction 
  y = y_train, # sequence we're predicting
  batch_size = batch_size, # how many samples to pass to our model at a time
  epochs = total_epochs, # how many times we'll look @ the whole dataset
  validation_split = 0.2) # how much data to hold out for testing as we go along

# how well did our trained model do?
trained_model

# plot how our model preformance changed during training 
plot(trained_model)


## -------------------------------------------------------------------------------------------
# Evaluate the model on the validation data
results <- model %>% keras::evaluate(X_test, y_test, verbose = 1)
results

pred_test = model %>% predict(X_test)
# print(postResample(pred = exp(pred_test), obs = testing$test_rate))
print(postResample(pred = pred_test, obs = testing$test_rate))

testing$baseline <- ((testing$sTest * testing$pState_popn) / testing$Tot_pop) * 1e3
print(postResample(pred <- testing$baseline, obs = testing$test_rate))

plot(y_test, pred_test[,1])
abline(0,1)

mydf = data.frame(obs = rep(testing$test_rate, 2), 
                  pred = c(testing$baseline, pred_test),
                  type = rep(c("base", "pred"), each = length(testing$test_rate)))
p1 = ggscatter(mydf, x = "obs", y = "pred", col = "type",
               main = paste0("COVID 19 testing (keras, raw-scale)")) + 
  geom_abline()
print(p1)

ggsave("./results/COVID19_keras_cv.pdf", p1)
