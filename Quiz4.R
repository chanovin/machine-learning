## Quiz 4, Machine Learning
##

## Question 1

library(ElemStatLearn)
library(caret)
data(vowel.train)
data(vowel.test)
vowel.test$y<-factor(vowel.test$y)
vowel.train$y <- factor(vowel.train$y)
set.seed(33833)
fit1<-train(y~.,method="rf",data=vowel.train)
fit2<-train(y~.,method="gbm",data=vowel.train)
predt1 <- predict(fit1,vowel.train)
predt2 <- predict(fit2, vowel.train)
pred1 <- predict(fit1,vowel.test)
pred2 <- predict(fit2, vowel.test)
sum(pred1==vowel.test$y)/length(pred1)
sum(pred2==vowel.test$y)/length(pred2)
agpredindex <- pred1==pred2
sum((pred1==vowel.test$y)[agpredindex])/sum(agpredindex)

## Question 2
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
set.seed(62433)
mod1 <- train(diagnosis~., data=training, method="rf")
mod2 <- train(diagnosis~., data=training, method="gbm")
mod3 <- train(diagnosis~., data=training, method="lda")
predtr1 <- predict(mod1, training)
predtr2 <- predict(mod2, training)
predtr3 <- predict(mod3, training)
predtrdf <- data.frame(predtr1,predtr2,predtr3,diagnosis=training$diagnosis)
modAgg <- train(diagnosis~., data=predtrdf, method="rf")
pred1 <- predict(mod1, testing)
pred2 <- predict(mod2, testing)
pred3 <- predict(mod3, testing)
preddf <- data.frame(predtr1=pred1, predtr2=pred2, predtr3=pred3)
predAgg <- predict(modAgg, preddf)
sum(pred1==testing$diagnosis)/length(testing$diagnosis)
sum(pred2==testing$diagnosis)/length(testing$diagnosis)
sum(pred3==testing$diagnosis)/length(testing$diagnosis)
sum(predAgg==testing$diagnosis)/length(testing$diagnosis)

## Question 3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(233)
mod <- train(CompressiveStrength~., data=concrete, method='lasso')
plot.enet(mod$finalModel, "penalty",TRUE)

## Question 4
library(lubridate) # For year() function below
dat = read.csv("gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)
mod <- bats(tstrain)
pred <- forecast(mod, level=95,h=length(testing$visitsTumblr))
sum(testing$visitsTumblr > pred$lower & testing$visitsTumblr < pred$upper)/length(testing$visitsTumblr)

## Question 5
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(325)
library(e1071)
mod <- svm(CompressiveStrength~., data=testing)
pred <- predict(mod, testing)
library(ModelMetrics)
rmse(testing$CompressiveStrength,pred)