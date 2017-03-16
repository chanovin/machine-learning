## Project Script File
## Coursera Machine Learning Course
## Author: Vincent Chan

training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
testing <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))

str(training)
## there appear to be a lot of NAs for the distribution data. are these data one observation per rep
## or is it one set of observations with temporally adjacent timestamp per rep?

## compare to the test set
str(testing)
## test set gives us 20 obs, so we need to determine the class of exercise based only on
## instantaneous data. we can discard all of the new_window data

## next to new_window (y/n) there is a num_window. probably corresponding to the rep.
## discard this for now.

## identify non-important variables
## user_name: should be independent of who does the exercise bc the movement is either correct or not
## raw_timestamp_part_1: time/date info should not affect quality of exercise
## raw_timestamp_part_2: see above
## cvtd_timestamp: see above
## new_window: indicates that a line is the completed rep summary info (we can recalculate this)
## any summary data for each rep (we can calculate this on our own if we need it)

## possible important variables:
## num_window: could be useful for performing summaries for each rep
## classe: outcome data
## all movement metrics data

## let's clean up the data somewhat
## drop rows that are new_windows
training <- training[which(training$new_window!="yes"),]
## drop IDed cols
training <- subset(training,select=-c(user_name, raw_timestamp_part_1, raw_timestamp_part_2,
                                      cvtd_timestamp, new_window))
## drop NA cols
training <- training[,colSums(is.na(training)) != nrow(training)]
## still need to drop kurtosis, skewness, max, min, amplitude
training <- subset(training,select=-c(kurtosis_roll_belt,kurtosis_picth_belt,kurtosis_yaw_belt,
                                      skewness_roll_belt,skewness_roll_belt.1,skewness_yaw_belt,
                                      max_yaw_belt,min_yaw_belt,amplitude_yaw_belt))
training <- subset(training,select=-c(kurtosis_roll_arm,
                                      kurtosis_picth_arm,kurtosis_yaw_arm,skewness_roll_arm,
                                      skewness_pitch_arm,skewness_yaw_arm))
training <- subset(training,select=-c(kurtosis_roll_dumbbell,
                                      kurtosis_picth_dumbbell,kurtosis_yaw_dumbbell,
                                      skewness_roll_dumbbell,skewness_pitch_dumbbell,
                                      skewness_yaw_dumbbell,max_yaw_dumbbell,min_yaw_dumbbell,
                                      amplitude_yaw_dumbbell))
training <-  subset(training,select=-c(kurtosis_roll_forearm,kurtosis_picth_forearm,
                                       kurtosis_yaw_forearm,skewness_roll_forearm,
                                       skewness_pitch_forearm,skewness_yaw_forearm,
                                       max_yaw_forearm,min_yaw_forearm, amplitude_yaw_forearm))
## lastly, X appears to be an index value
training <- subset(training,select=-X)
## now we should be down to the usable variables

## split the data set for cross-validation
## our training set is rather large, so we'll use 1/2 of it for validation and 1/2 for training
library(caret)
library(gbm)
library(AppliedPredictiveModeling)

set.seed(314159)
indexTrain = createDataPartition(training$classe, p = 1/2)[[1]]
trainset = training[ indexTrain,]
valset = training[-indexTrain,]

## create three models
mod1 <- train(classe~.,data=trainset,method="gbm")
mod2 <- train(classe~.,data=trainset,method="rf")
mod3 <- train(classe~.,data=trainset,method="lda")
## check the models vs actual values for reasonable in-sample accuracy
predTr1 <- predict(mod1, trainset)
predTr2 <- predict(mod2, trainset)
predTr3 <- predict(mod3, trainset)
table(predTr1,trainset$classe)
sum(predTr1==trainset$classe)/length(trainset$classe)
table(predTr2,trainset$classe)
sum(predTr2==trainset$classe)/length(trainset1$classe)
table(predTr3,trainset$classe)
sum(predTr3==trainset$classe)/length(trainset$classe)

## 2 is best with 1 close and 3 doing poorly

## check the models for out-of-sample accuracy against our valset
predval1 <- predict(mod1,valset)
predval2 <- predict(mod2, valset)
predval3 <- predict(mod3, valset)
table(predval1, valset$classe)
sum(predval1==valset$classe)/length(valset$classe)
table(predval2, valset$classe)
sum(predval2==valset$classe)/length(valset$classe)
table(predval3, valset$classe)
sum(predval3==valset$classe)/length(valset$classe)

## 2 continues to outperform 1 by just a touch
## we'll apply it to the test set now
predtest2 <- predict(mod2,testing)