
---
title: "Quality of Activity --- Machine Learning Project"
output:
  html_document:
    keep_md: true
---

## Objective

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now
possible to collect a large amount of data about personal activity relatively
inexpensively.
These type of devices are part of the quantified self movement --- a group of
enthusiasts who take measurements about themselves regularly to improve their
health, to find patterns in their behavior, or because they are tech geeks.
One thing that people regularly do is quantify how much of a particular activity
they do, but they rarely quantify how well they do it.
In this project, the manner in which six participants do their exercises is
predicted using data from accelerometers on the belt, forearm, arm, and dumbell.
The manner they do their exercises is identified as `classe` A to E (A is
correct execution of the exercise, B to E are common mistakes).

Using the given data, an algorithm using the sensors' readings is developed to
predict the A to E classes.

## Data Processing

### Set libraries used in the analysis


```r
library(tidyverse)
library(caret)
```

### Load the data


```r
train_url <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train_file <- "./data/pml-training.csv"
test_file  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(train_file)) {
  download.file(train_url, destfile = train_file)
}
if (!file.exists(test_file)) {
  download.file(test_url, destfile = test_file)
}

train <- read.csv("./data/pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test <- read.csv("./data/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
```

### Data processing

Only variables related to belt, forearm, arm, or dumbbell are kept.
Variable with with near-zero variance and those with missing values are also
removed from the data sets.


```r
sensors <- grep(pattern = "_belt|_forearm|_arm|_dumbbell", names(train))
train <- train[, c(sensors, 160)]

train_nzv <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[, train_nzv$nzv == FALSE]

train <- train[, colSums(is.na(train)) == 0]
test <- test[, names(test) %in% c(names(train), "problem_id")]
```

### Partitioning the training set

The processed training set is partitioned into a training data set (60%) and a
validation data set (40%).
The validation data set will be used for cross-validation.


```r
set.seed(123)
in_train <- createDataPartition(y = train$classe, p = 0.6, list = FALSE)
training <- train[in_train, ]
testing <- train[-in_train, ]
```

## Modelling

A predictive model using Random Forest algorithm was fit with 3-fold cross
validation.
Variables are also centered and scaled.


```r
set.seed(123)
ctrl_rf <- trainControl(method = "cv", 3)
model_rf <- train(classe ~ .,
                  method = "rf",
                  preProcess = c("center", "scale"),
                  trControl = ctrl_rf,
                  data = training)
model_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.89%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3343    3    0    0    2 0.001493429
## B   14 2260    5    0    0 0.008336990
## C    0   27 2022    5    0 0.015579357
## D    1    0   39 1889    1 0.021243523
## E    1    0    1    6 2157 0.003695150
```

Model's performance is then assessed on the validation data set.


```r
predict_rf <- predict(model_rf, testing)
confusionMatrix(testing$classe, predict_rf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    2    0    1    0
##          B   13 1501    4    0    0
##          C    0   11 1355    2    0
##          D    0    0   21 1264    1
##          E    0    0    2    3 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9924          
##                  95% CI : (0.9902, 0.9942)
##     No Information Rate : 0.2858          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9903          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9942   0.9914   0.9805   0.9953   0.9993
## Specificity            0.9995   0.9973   0.9980   0.9967   0.9992
## Pos Pred Value         0.9987   0.9888   0.9905   0.9829   0.9965
## Neg Pred Value         0.9977   0.9979   0.9958   0.9991   0.9998
## Prevalence             0.2858   0.1930   0.1761   0.1619   0.1833
## Detection Rate         0.2841   0.1913   0.1727   0.1611   0.1832
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9968   0.9944   0.9892   0.9960   0.9993
```

```r
## accuracy
postResample(predict_rf, testing$classe)
```

```
##  Accuracy     Kappa 
## 0.9923528 0.9903255
```

```r
## Out-of-sample error
1 - as.numeric(confusionMatrix(testing$classe, predict_rf)$overall[1])
```

```
## [1] 0.007647209
```

Model's accuracy is 99.24%
and the out-of-sample error is 0.76%.

## Predicting for Test data set


```r
predict(model_rf, test)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
