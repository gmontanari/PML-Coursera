Practical Machine Learning: Mauricio G. Tec
========================================================



In this project we use a Random Forest to ---- data from ______ to try to predict ______. For the purposes of this html demonstation, instead of taking the full training set for training, I will split it in a training testing set, so that I can compare and tabulate the predicted values against the real ones in the training set. 

However, I use the full model with 3-fold cross validation to compute the final results in the project (Note: I use only 3-fold validation since it is highly demanding in computational power).

Explanation of the data here ----------

* Set-up

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.0.3
```

```r
library(ggplot2)
library(gridExtra)
```

```
## Warning: package 'gridExtra' was built under R version 3.0.3
```

```
## Loading required package: grid
```

```r
set.seed(110104)

training <- read.csv("pml-training.csv")[ ,-1]
# remove columns with NA's or empty values
training <- training[ ,!sapply(training, function(x) 
  any(is.na(x) | (x=="")))] 
testing <- read.csv("pml-testing.csv")[ ,-1]
# matching the variables in testing and training sets
vars <- names(training) 
testing <- testing[ ,names(testing) %in% vars]
```

* 1) Splitting for demonstation purposes
========================================
========================================



```r
inTraining <- createDataPartition(training$classe, p = .6, list = FALSE)
training.sub <- training[inTraining, ]
testing.sub <- training[-inTraining, ]
# For demonstration purposes we only use 250 observations
training.sub <- training.sub[sample(dim(training.sub)[1], 50), ]
```


```r
modFit.dem <- train(classe ~., data=training.sub, method="rf", prox=TRUE)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## Loading required namespace: e1071
```

```
## Warning: model fit failed for Resample21: mtry= 2 Error in randomForest.default(x, y, mtry = param$mtry, ...) : 
##   Can't have empty classes in y.
## 
## Warning: model fit failed for Resample21: mtry=41 Error in randomForest.default(x, y, mtry = param$mtry, ...) : 
##   Can't have empty classes in y.
## 
## Warning: model fit failed for Resample21: mtry=80 Error in randomForest.default(x, y, mtry = param$mtry, ...) : 
##   Can't have empty classes in y.
## 
## Warning: There were missing values in resampled performance measures.
```

```r
modFit.dem
```

```
## Random Forest 
## 
## 50 samples
## 58 predictors
##  5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 50, 50, 50, 50, 50, 50, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##    2    0.4101    0.2141  0.1223       0.1472  
##   41    0.3524    0.1503  0.1063       0.1193  
##   80    0.3315    0.1303  0.1038       0.1139  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```


```r
pred <- predict(modFit.dem, testing.sub)
testingTRUE <- testing.sub$classe
predRight <- pred==testingTRUE
table(pred, testingTRUE)
```

```
##     testingTRUE
## pred    A    B    C    D    E
##    A 1451  773  783  725  656
##    B   90  455  145   83  224
##    C    1    2   54    0    1
##    D    0   23   63  143    8
##    E  690  265  323  335  553
```


Here we see an example of a tree

```r
tree <- getTree(modFit.dem$finalModel, k=2, labelVar=TRUE)
tree
```

```
##    left daughter right daughter                      split var split point
## 1              2              3               gyros_dumbbell_x       0.055
## 2              4              5               user_namecharles       0.500
## 3              6              7                   accel_belt_y      26.500
## 4              8              9                    accel_arm_y     -57.000
## 5             10             11                   roll_forearm      -9.250
## 6             12             13                gyros_forearm_x      -0.120
## 7             14             15               gyros_dumbbell_y      -0.095
## 8              0              0                           <NA>       0.000
## 9             16             17                   accel_belt_z      31.000
## 10             0              0                           <NA>       0.000
## 11             0              0                           <NA>       0.000
## 12            18             19                gyros_forearm_x      -0.370
## 13             0              0                           <NA>       0.000
## 14            20             21               gyros_dumbbell_y      -0.265
## 15            22             23                    yaw_forearm     -17.450
## 16            24             25               magnet_forearm_x    -382.000
## 17             0              0                           <NA>       0.000
## 18             0              0                           <NA>       0.000
## 19             0              0                           <NA>       0.000
## 20             0              0                           <NA>       0.000
## 21            26             27              magnet_dumbbell_y       7.500
## 22            28             29                      pitch_arm     -13.950
## 23            30             31                total_accel_arm      27.500
## 24             0              0                           <NA>       0.000
## 25             0              0                           <NA>       0.000
## 26            32             33               magnet_forearm_z     580.000
## 27             0              0                           <NA>       0.000
## 28             0              0                           <NA>       0.000
## 29             0              0                           <NA>       0.000
## 30             0              0                           <NA>       0.000
## 31            34             35 cvtd_timestamp02/12/2011 13:34       0.500
## 32             0              0                           <NA>       0.000
## 33             0              0                           <NA>       0.000
## 34             0              0                           <NA>       0.000
## 35             0              0                           <NA>       0.000
##    status prediction
## 1       1       <NA>
## 2       1       <NA>
## 3       1       <NA>
## 4       1       <NA>
## 5       1       <NA>
## 6       1       <NA>
## 7       1       <NA>
## 8      -1          B
## 9       1       <NA>
## 10     -1          A
## 11     -1          B
## 12      1       <NA>
## 13     -1          B
## 14      1       <NA>
## 15      1       <NA>
## 16      1       <NA>
## 17     -1          B
## 18     -1          E
## 19     -1          D
## 20     -1          E
## 21      1       <NA>
## 22      1       <NA>
## 23      1       <NA>
## 24     -1          A
## 25     -1          D
## 26      1       <NA>
## 27     -1          E
## 28     -1          A
## 29     -1          E
## 30     -1          A
## 31      1       <NA>
## 32     -1          D
## 33     -1          A
## 34     -1          A
## 35     -1          D
```


```r
classvars <- as.character(tree[ ,"split var"])[1:4]
q1 <- qplot(testing.sub[ ,classvars[1]], testing.sub[ ,classvars[2]], data=testing.sub, main="new data predictions", xlab=classvars[1], ylab=classvars[2], colour=predRight)
q2 <- qplot(testing.sub[ ,classvars[3]], testing.sub[ ,classvars[4]], data=testing.sub, main="new data predictions", xlab=classvars[3], ylab=classvars[4], colour=predRight)
grid.arrange(q1, q2, ncol=2)
```

```
## Error: undefined columns selected
```


* Model with cross-validation and final prediction

First we set-up the controls for the cross validation. We will do 10-fold cross-validation but only three repetitions since I lack the computational power. 


```r
# 10-fold cross validation (gives a better estimation of the error)
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated two times
                           repeats = 3)
```

Now we run the model. We can observer that even when restricting to 2500 out of the 19622 available observations it is really slow.


```r
# Again, doing the predictions with all the database is unfeasable, here we will take 2500 observations.
# ptm <- proc.time()
#modFit <- train(classe ~., data=training[sample(dim(training)[1], 2500), ],  trControl = fitControl, method="rf", prox=TRUE)
#modFit
#proc.time() - ptm
```

Here is the list of prediction of the 20 individuals in the testing set.


```r
#predict(modFit, testing)
```


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
#pml_write_files(predict(modFit, testing))
```

