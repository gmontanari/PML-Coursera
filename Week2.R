rm(list=ls(all=TRUE))

# EXERCISE 1
library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50, list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]


# EXERCISE 2
library(Hmisc)
data(concrete)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
cutCS <- cut2(training$CompressiveStrength)
qplot(1:nrow(training), CompressiveStrength, data=training, colour=cutCS)

cutFS <- cut2(training$FlyAsh)
qplot(1:nrow(training), CompressiveStrength, data=training, colour=cutFS)

cutAge <- cut2(training$Age)
qplot(1:nrow(training), CompressiveStrength, data=training, colour=cutAge)


# EXERCISE 3
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
hist(log(training$Superplasticizer+1), breaks=50)


# EXERCISE 4
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
ILvars <- training[ , grep("\\bIL", names(adData))]
preProc <- preProcess(ILvars, method="pca")
summary(preProc)
library(FactoMineR)
result <- PCA(ILvars)
result$eig


# EXERCISE 5
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
trainPC <- data.frame(diagnosis=training$diagnosis, training[ , grep("\\bIL", names(adData))])
modelFit <- train(trainPC$diagnosis ~ ., method= "glm", preProcess="pca", data=trainPC)
confusionMatrix(testing$diagnosis, predict(modelFit, testing))
modelFit2 <- train(trainPC$diagnosis ~ ., method= "glm", data=trainPC)
confusionMatrix(testing$diagnosis, predict(modelFit2, testing))
