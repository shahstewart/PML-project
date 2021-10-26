---
title: Model selection for project 'Predicting Barbell Lift Quality From Accelerometer Data'
author: Sangeeta Shah
date: Oct 24, 2021
output:
    html_document:
        keep_md: yes
---



&nbsp;  

This page contains the test models, the code and the model selection process details for the accompanying project
[_**Predicting Barbell Lift Quality from Accelerometer Data**_][1] 

&nbsp;  

## Data preprocessing

```r
library(caret); library(dplyr)

trainingSet <- read.csv('pml-training.csv'); testingSet <- read.csv('pml-testing.csv')

# remove irrelevant columns
trainingSet <- trainingSet[, -(1:7)]; testingSet <- testingSet[, -(1:7)]

# remove columns with missing / invalid data
badCols <- colSums(is.na(trainingSet) | trainingSet == '' | trainingSet == '#DIV/0!')
badCols <- badCols[badCols > (dim(trainingSet)[1] * .95)]
library(dplyr)
trainingSet <- trainingSet %>% select(-all_of(names(badCols)))
testingSet <- testingSet %>%  select(-all_of(names(badCols)))

# remove highly correlated columns
library(caret)
collinear <- findCorrelation(cor(trainingSet[, -53]), cutoff=0.95)
trainingSet <- trainingSet[, -collinear]; testingSet <- testingSet[, -collinear]


# factor the output column
trainingSet$classe <- factor(trainingSet$classe)

## divide the data into training and validation sets
set.seed(345325)
inTrain <- createDataPartition(y=trainingSet$classe, p=0.80, list=F)
trainingSet <- trainingSet[inTrain,]; validationSet <- trainingSet[-inTrain, ]
```

&nbsp;  

## Test Models

### Random Forests

#### Random Forests [ntree: 100, metric:Accuracy]

```r
set.seed(3333)
seedV <- vector(mode = 'list', length=2); seedV[[1]] <- sample.int(n=1000, 7); seedV[[2]] <- sample.int(n=1000, 1)
tuningGrid <- data.frame(mtry=6:12)

startTime <- proc.time()
mrf100 <- train(classe ~ ., data=trainingSet, method='rf', nodesize=5, 
                  metric='Accuracy', importance=T,
                  trControl= trainControl(method='oob', allowParallel=T, seeds=seedV),
                  tuneGrid= tuningGrid, ntree=100,
                  allowParallel=T)
mrf100time <- proc.time() - startTime
mrf100
```

```
## Random Forest 
## 
## 15699 samples
##    48 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    6    0.9927384  0.9908139
##    7    0.9938850  0.9922645
##    8    0.9928021  0.9908950
##    9    0.9932480  0.9914589
##   10    0.9946493  0.9932318
##   11    0.9943945  0.9929094
##   12    0.9942035  0.9926675
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 10.
```

&nbsp;  


#### Random Forests [ntree: 150, metric:Accuracy]

```r
set.seed(3333)
startTime <- proc.time()
mrf150 <- train(classe ~ ., data=trainingSet, method='rf', nodesize=5, 
                  metric='Accuracy', importance=T,
                  trControl= trainControl(method='oob', allowParallel=T, seeds=seedV),
                  tuneGrid= tuningGrid, ntree=150,
                  allowParallel=T)
mrf150time <- proc.time() - startTime
mrf150
```

```
## Random Forest 
## 
## 15699 samples
##    48 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    6    0.9938213  0.9921839
##    7    0.9942035  0.9926676
##    8    0.9938213  0.9921841
##    9    0.9940761  0.9925065
##   10    0.9947767  0.9933929
##   11    0.9943945  0.9929094
##   12    0.9946493  0.9932318
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 10.
```
&nbsp;  

#### Random Forests [ntree: 200, metric:Accuracy]

```r
set.seed(3333)
startTime <- proc.time()
mrf200 <- train(classe ~ ., data=trainingSet, method='rf', nodesize=5, 
                  metric='Accuracy', importance=T,
                  trControl= trainControl(method='oob', allowParallel=T, seeds=seedV),
                  tuneGrid= tuningGrid, ntree=200,
                  allowParallel=T)
mrf200time <- proc.time() - startTime
mrf200
```

```
## Random Forest 
## 
## 15699 samples
##    48 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    6    0.9939487  0.9923451
##    7    0.9942035  0.9926675
##    8    0.9942035  0.9926675
##    9    0.9936939  0.9920228
##   10    0.9948404  0.9934735
##   11    0.9950315  0.9937151
##   12    0.9947767  0.9933930
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 11.
```


&nbsp;  

### Gradient Boosting 
#### [interaction depth: 1:3, ntree: 50:150, shrinkage: 0.1]


```r
tuningGrid <- expand.grid(
    interaction.depth= 1:3,
    n.trees = (1:3)*50,
    shrinkage = .1,
    n.minobsinnode = 10)

set.seed(3333)
seedV <- vector(mode = 'list', length=51); 
for(i in 1:50) seedV[[i]] <- sample.int(n=1000, 400); seedV[[51]] <- sample.int(n=1000, 1)

startTime <- proc.time()
mgbm <- train(classe ~ ., data=trainingSet, method='gbm', metric='Accuracy', 
                   trControl= trainControl(method='repeatedcv', repeats=5, allowParallel=T, seeds=seedV),
                   tuneGrid=tuningGrid,  verbose=F)
mgbmTime <- proc.time() - startTime
mgbm
```

```
## Stochastic Gradient Boosting 
## 
## 15699 samples
##    48 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## Summary of sample sizes: 14129, 14130, 14128, 14128, 14131, 14130, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7388364  0.6689213
##   1                  100      0.8120013  0.7620552
##   1                  150      0.8456335  0.8046150
##   2                   50      0.8496209  0.8094589
##   2                  100      0.9028847  0.8770850
##   2                  150      0.9278675  0.9087181
##   3                   50      0.8926426  0.8640696
##   3                  100      0.9400212  0.9240919
##   3                  150      0.9599584  0.9493371
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

## Testing the Models
**Model mrf100**

```r
mrf100Preds <- predict(mrf100$finalModel, validationSet)
crf100 <- confusionMatrix(mrf100Preds, validationSet$classe)
mrf100Accuracy <- round(sum(mrf100Preds == validationSet$classe) / nrow(validationSet), digits= 2)
mrf100Error <- 1 - mrf100Accuracy
crf100
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 891   0   0   0   0
##          B   0 604   0   0   0
##          C   0   0 574   0   0
##          D   0   0   0 491   0
##          E   0   0   0   0 574
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1927   0.1832   0.1567   0.1832
## Detection Rate         0.2843   0.1927   0.1832   0.1567   0.1832
## Detection Prevalence   0.2843   0.1927   0.1832   0.1567   0.1832
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
&nbsp;  
**Model mrf150**

```r
mrf150Preds <- predict(mrf150$finalModel, validationSet)
crf150 <- confusionMatrix(mrf150Preds, validationSet$classe)
mrf150Accuracy <- round(sum(mrf150Preds == validationSet$classe) / nrow(validationSet), digits= 2)
mrf150Error <- 1 - mrf150Accuracy
crf150
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 891   0   0   0   0
##          B   0 604   0   0   0
##          C   0   0 574   0   0
##          D   0   0   0 491   0
##          E   0   0   0   0 574
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1927   0.1832   0.1567   0.1832
## Detection Rate         0.2843   0.1927   0.1832   0.1567   0.1832
## Detection Prevalence   0.2843   0.1927   0.1832   0.1567   0.1832
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
&nbsp;    
**Model mrf200**

```r
mrf200Preds <- predict(mrf200$finalModel, validationSet)
crf200 <- confusionMatrix(mrf200Preds, validationSet$classe)
mrf200Accuracy <- round(sum(mrf200Preds == validationSet$classe) / nrow(validationSet), digits= 2)
mrf200Error <- 1 - mrf200Accuracy
crf200
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 891   0   0   0   0
##          B   0 604   0   0   0
##          C   0   0 574   0   0
##          D   0   0   0 491   0
##          E   0   0   0   0 574
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1927   0.1832   0.1567   0.1832
## Detection Rate         0.2843   0.1927   0.1832   0.1567   0.1832
## Detection Prevalence   0.2843   0.1927   0.1832   0.1567   0.1832
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
&nbsp;  

**Model mgmb**

```r
mgbmPreds <- predict(mgbm, validationSet)
cmgbm <- confusionMatrix(mgbmPreds, validationSet$classe)
mgbmAccuracy <- round(sum(mgbmPreds == validationSet$classe) / nrow(validationSet), digits= 2)
mgbmError <- 1 - mgbmAccuracy
cmgbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 882   8   0   1   2
##          B   6 586  11   3   5
##          C   2   8 559  11   7
##          D   1   0   4 471   3
##          E   0   2   0   5 557
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9748        
##                  95% CI : (0.9687, 0.98)
##     No Information Rate : 0.2843        
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.9681        
##                                         
##  Mcnemar's Test P-Value : 0.03108       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9899   0.9702   0.9739   0.9593   0.9704
## Specificity            0.9951   0.9901   0.9891   0.9970   0.9973
## Pos Pred Value         0.9877   0.9591   0.9523   0.9833   0.9876
## Neg Pred Value         0.9960   0.9929   0.9941   0.9925   0.9934
## Prevalence             0.2843   0.1927   0.1832   0.1567   0.1832
## Detection Rate         0.2814   0.1870   0.1784   0.1503   0.1777
## Detection Prevalence   0.2849   0.1950   0.1873   0.1528   0.1800
## Balanced Accuracy      0.9925   0.9802   0.9815   0.9781   0.9838
```

&nbsp;  

## Selecting the model

```r
mbest <- mgbm$bestTune
mrf100M <- c(Algorithm = 'Random Forest', ntree = 100, mtry = mrf100$bestTune['mtry'], shrinkage = NA,
             interaction.depth = NA, Metric = 'Accuracy', Accuracy = '99.46%',
             Compute.Time = round(mrf100time['elapsed'], 2), Validation.Accuracy = round(mrf100Accuracy, 2),
             Misclassification.Error = round(mrf100Error, 2))
mrf150M <- c(Algorithm = 'Random Forest', ntree = 150, mtry = mrf150$bestTune['mtry'], shrinkage = NA,
             interaction.depth = NA, Metric = 'Accuracy', Accuracy = '99.48%',
             Compute.Time = mrf150time['elapsed'], Validation.Accuracy = round(mrf150Accuracy, 2),
             Misclassification.Error = round(mrf150Error, 2))
mrf200M <- c(Algorithm = 'Random Forest', ntree = 200, mtry = mrf200$bestTune['mtry'], shrinkage = NA,
             interaction.depth = NA, Metric = 'Accuracy', Accuracy = '99.50%', 
             Compute.Time = mrf200time['elapsed'], Validation.Accuracy = round(mrf200Accuracy, 2),
             Misclassification.Error = round(mrf200Error, 2))
mgbmM <-   c(Algorithm = 'Gradient Boosting', ntree = mbest['n.trees'], mtry = NA, shrinkage = mbest['shrinkage'],
             interaction.depth = mbest['interaction.depth'], Metric = 'Accuracy', Accuracy = '96.00%',
             Compute.Time = mgbmTime['elapsed'], Validation.Accuracy = round(mgbmAccuracy, 2),
             Misclassification.Error = round(mgbmError, 2))

models <- cbind(mrf100M, mrf150M, mrf200M, mgbmM)
colnames(models) <- c('Rf ntree 100', 'Rf ntree 150', 'Rf ntree 200', 'Gradient Boosting')
kable(models)
```



|                        |Rf ntree 100  |Rf ntree 150  |Rf ntree 200  |Gradient Boosting |
|:-----------------------|:-------------|:-------------|:-------------|:-----------------|
|Algorithm               |Random Forest |Random Forest |Random Forest |Gradient Boosting |
|ntree                   |100           |150           |200           |150               |
|mtry.mtry               |10            |10            |11            |NA                |
|shrinkage               |NA            |NA            |NA            |0.1               |
|interaction.depth       |NA            |NA            |NA            |3                 |
|Metric                  |Accuracy      |Accuracy      |Accuracy      |Accuracy          |
|Accuracy                |99.46%        |99.48%        |99.50%        |96.00%            |
|Compute.Time.elapsed    |96.34         |141.64        |188.92        |2056.04           |
|Validation.Accuracy     |1             |1             |1             |0.97              |
|Misclassification.Error |0             |0             |0             |0.03              |
** In the table above, while _Accuracy_ is the accuracy of the model as provided by the train function,
_Validation.Accuracy_ is calculated based on predictions obtained on the validation set.


[1]: project.html  
