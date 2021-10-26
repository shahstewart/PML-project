---
title: Predicting Barbell Lift Quality From Accelerometer Data
subtitle: Coursera-JHU Practical Machine Learning Course Project
author: Sangeeta Shah
date: Oct 24, 2021
output:
    html_document:
        keep_md: yes
---


&nbsp;  

## Executive Summary
The provided data has been [collected from accelerometers][5] in devices like Jawbone Up, Nike FuelBand and Fitbit, 
mounted on the belt, forearm, arm, and dumbbell of 6 participants performing barbell 
lifts correctly or incorrectly in 5 different ways. The goal of this project is to develop, using the given data,
a model to predict which of the 5 _ways_ a barbell lift was performed.

The following steps were carried out to develop a model for prediction.  
1. retrieve the data from provided sources.  
2. Explore the data for any obvious anomalies/ relationships 
3. prepare the data by removing missing, redundant, unnecessary features.  
4. Split the training data into _training_ and _validation_ sets.  
5. Fit various models on the training set and validate them using the validation set to select the best fit.  

Three _Random Forest_ and one _Gradient Boosting_ models were created and tested. A _Random Forest_ model with 
$ntree$ value of 200 ($mrf200$) was selected based on accuracy and out of sample error rate. 
Predictions regarding the type of barbell lift were made on the test data using $mrf200$.  

&nbsp;  

## The data
The [The project description page][1] provides links to the [training][2] and [testing][3] data sets. The data was 
retrieved from these sources. 


```r
if (!file.exists('pml-training.csv')) {
    url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'  
    download.file(url, 'pml-training.csv')
}

if (!file.exists('pml-testing.csv')) {
    url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'  
    download.file(url, 'pml-testing.csv')
}
trainingSet <- read.csv('pml-training.csv'); testingSet <- read.csv('pml-testing.csv')
c(trainingSet=dim(trainingSet), testingSet=dim(testingSet))
```

```
## trainingSet1 trainingSet2  testingSet1  testingSet2 
##        19622          160           20          160
```

```r
table(trainingSet$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

&nbsp;  
  
## Preprocessing data  
The training and testing datasets contain 19622 and 20 records respectively, each with 160 columns. The column
_**classe**_ holds the record of how the lift was performed. Values 'A' through 'E' correspond to the 5 different
ways the lift was performed.  

159 features is a lot of features, and quite likely to include irrelevant and correlated data, that 
will lead to increased variance. The data was explored to reduce the number of features without losing information.
Upon manual inspection, it is clear that the first 7 columns of the data hold information such as record id, subject
name, timestamp, which is irrelevant to our analysis. These columns were removed.  


```r
head(names(trainingSet), 8)
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"           "roll_belt"
```

```r
trainingSet <- trainingSet[, -(1:7)]; testingSet <- testingSet[, -(1:7)]
```
&nbsp;  

We also see a large number of columns that mainly hold empty data or NA or string '#Div/0!'. 
If most of their data is missing or invalid, these columns are highly unlikely to contribute any meaningful 
influence on the outcome. Therefore, columns with more than 95% missing or invalid values were removed

```r
badCols <- colSums(is.na(trainingSet) | trainingSet == '' | trainingSet == '#DIV/0!')
badCols <- badCols[badCols > (dim(trainingSet)[1] * .95)] # 100 
library(dplyr)
trainingSet <- trainingSet %>% select(-all_of(names(badCols)))
testingSet <- testingSet %>%  select(-all_of(names(badCols)))
```
&nbsp;  

Let's also see if there are any columns with near-zero variance and any columns that are highly correlated. We will
remove if we find any.

```r
library(caret)
unchanging <- nearZeroVar(trainingSet) # 0
collinear <- findCorrelation(cor(trainingSet[, -53]), cutoff=0.95) # 4 columns
trainingSet <- trainingSet[, -collinear]; testingSet <- testingSet[, -collinear]
```
&nbsp;  
The _classe_ column, which is the output column, was converted into a factor. This will let the model training 
functions know that we are trying to build a classification model, rather than a regression model.

```r
trainingSet$classe <- factor(trainingSet$classe)
c(trainingSet=dim(trainingSet), testingSet=dim(testingSet))
```

```
## trainingSet1 trainingSet2  testingSet1  testingSet2 
##        19622           49           20           49
```
The raw data contains 159 features. After preprocessing, we are left with 48 features.  

&nbsp;  

## Splitting the training data
We have a training and a testing set that we processed similarly. Now, we will keep the testing set aside and divide
the training set into training and validation sets. While the training set will be used for training the models, the
validation data will be used for testing these models. Since the validation set is not used for training, fitting 
models to this set will give us a good estimate of out of sample error rates. 

```r
set.seed(345325)
inTrain <- createDataPartition(y=trainingSet$classe, p=0.80, list=F)
trainingSet <- trainingSet[inTrain,]; validationSet <- trainingSet[-inTrain, ]
c(trainingSet= dim(trainingSet), validationSet= dim(validationSet))
```

```
##   trainingSet1   trainingSet2 validationSet1 validationSet2 
##          15699             49           3134             49
```

&nbsp;  

## Models
We are trying to predict which of the 5 ways the barbell lift has been performed based on the numeric data from
the accelerometers. This is a multi-class classification problem. _Random Forest_ and _Gradient Boosting_ could be
good algorithms for this type of problem. Due to space constrain, all code for model fitting is in put in the 
accompanying file [testModels.html][7].

### Random Forest Models
**Model tuning**  
3 random forest models with $ntree$ values of 100, 150 and 200 were built. The caret package sets the default for $ntree$ 
to 500. However, this can cause performance issues on PCs. Low ntree values can lead to overfitting. The 3 models will
allow us to pick a suitable $ntree$ value.  

The default for the $mtry$ parameter in caret for random forest is the square root of the total number
of features, ~ 7 in our case (48 features). Fine-tuning **mtry** parameter may be useful in reducing 
overfitting. **mtry** values of 6 to 12 were tried.  

While accuracy was used as the selection metric, out of bag error (oob) was used for cross-validation and
parallel processing was allowed for increased computation efficiency.
&nbsp;  

### Gradient Boosting Model
Gradient boosting are computation-intensive and take a long time to run on personal computers, this puts some
constraints on tuning parameters. Considering long computation times required for low $shrinkage$ values, specially
for a large dataset like this one, the $shrinkage$ parameter was set at 0.1. $Interaction depth$ values of 1:3 and 
$ntree$ values of 50:150 were tried. The cross-validation method was _repeatedcv_. 

&nbsp;  

## Model Selection 
A total of 4 models were built, 3 random forest models _mrf100_, _mrf150_, and _mrf200_, and 
one gradient boosting model, _mgbm_. Details of all four models can be seen on the [testModels page][7].  

All three random forest models have accuracy between 99.46 to 99.5%. While the gradient boosting models has an
accuracy of 96%. Compared to the random forest models, _mgbm_ takes 20-30 times more time to compute.

The models were tested on the validation dataset and confusion matrices outputted (please see [testModels page][7])
to calculate prediction accuracy and misclassification error. Below is a table showing comparison of these 4 models.  

|                    |Rf ntree 100  |Rf ntree 150  |Rf ntree 200  |Gradient Boosting |
|:-------------------|:-------------|:-------------|:-------------|:-----------------|
|Algorithm           |Random Forest |Random Forest |Random Forest |Gradient Boosting |
|ntree               |100           |150           |200           |150               |
|mtry                |10            |10            |11            |NA                |
|shrinkage           |NA            |NA            |NA            |0.1               |
|interaction.depth   |NA            |NA            |NA            |3                 |
|Metric              |Accuracy      |Accuracy      |Accuracy      |Accuracy          |
|Accuracy            |99.46%        |99.48%        |99.50%        |96.00%            |
|Compute.Time        |94.53         |139.98        |187.58        |2021.55           |
|Prediction.Accuracy |1             |1             |1             |0.97              |
|Oos.Misclass.Error  |0             |0             |0             |0.3               |
<hr style='margin-bottom:0' />

**Notes:**   
1. The code for this table could be seen in the [testModels page][7]  
2. The row _Accuracy_ shows the values of model accuracy as provided by the _caret::train_ function.  
3. The row _Prediction.Accuracy_ refers to the accuracy of validation set predictions made by the model.
4. The field _Oos.Misclass.Error_ holds the out of sample misclassification error.  

&nbsp;  
All 3 random forest models show higher accuracy than the $mgbm$ model, and they all predict the outcome with
100% accuracy in the validation dataset indicating 0% out of sample error rate. We will select 
$rf200$ as our final model because it has the highest accuracy and because higher $ntree$ values may 
help reduce overfitting.

&nbsp;  

## The Selected Model: Model mrf200

```r
set.seed(3333)
seedV <- vector(mode = 'list', length=2); seedV[[1]] <- sample.int(n=1000, 7); seedV[[2]] <- sample.int(n=1000, 1)
tuningGrid <- data.frame(mtry=6:12)
mrf200 <- train(classe ~ ., data=trainingSet, method='rf', nodesize=5, 
                  metric='Accuracy', importance=T,
                  trControl= trainControl(method='oob', allowParallel=T, seeds=seedV),
                  tuneGrid= tuningGrid, ntree=200,
                  allowParallel=T)
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

```r
plot(mrf200)
```

![](D:/dev/R/dsCourseWorkProjects/PMLCourseProject/project_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

```r
varImp(mrf200)
```

```
## rf variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 48)
## 
##                           A     B     C     D     E
## yaw_belt             100.00 75.62 79.56 91.98 62.68
## pitch_belt            36.30 88.83 62.67 53.30 49.18
## magnet_dumbbell_z     74.11 62.78 79.17 65.48 58.72
## magnet_dumbbell_y     50.83 50.30 63.76 54.08 44.11
## pitch_forearm         47.58 52.74 61.42 48.02 55.08
## gyros_belt_z          26.41 40.65 35.56 28.19 37.03
## magnet_belt_z         25.35 28.89 27.06 40.00 30.37
## magnet_belt_y         31.07 38.38 37.80 38.27 33.00
## roll_forearm          34.12 28.58 38.22 23.12 28.74
## magnet_belt_x         13.90 35.84 33.02 23.81 27.64
## roll_dumbbell         26.78 33.90 32.80 35.80 30.55
## gyros_forearm_y       12.89 35.72 29.44 29.82 22.57
## gyros_dumbbell_y      22.50 18.04 35.67 20.26 14.34
## roll_arm              18.79 34.91 25.32 34.55 23.66
## accel_forearm_x       17.11 28.70 25.26 34.26 24.89
## total_accel_dumbbell  18.19 27.54 24.04 22.44 34.12
## accel_dumbbell_y      32.73 31.29 31.60 26.96 33.88
## gyros_arm_x           15.20 33.78 20.54 16.24 21.25
## magnet_arm_z          18.45 32.74 22.03 23.13 18.82
## magnet_forearm_z      26.46 29.77 23.87 32.63 29.32
```
The table above lists top 20 (out of 48) features that are most important in outcome prediction for this model.  

&nbsp;  

## Validation Set Predictions
Here are the results of predicting outcome of the validation data using $mrf200$:

```r
predictions <- predict(mrf200, validationSet)
confusionMatrix(predictions, validationSet$classe)
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

## Predicting the Test Data
Let's now predict the _classe_ values for the test dataset using model $mrf200$.  

```r
predictions <- predict (mrf200, testingSet)
predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

&nbsp;  

## Conclusions
Since this project presented with a classification problem, algorithms _Random Forests_ and _Gradient Boosting_,
which are well suited to multi-class classification were used.

using the $caret$ package's $train$ function, several models with high prediction accuracy could be build for 
predicting the way a barbell lift was performed. Training using caret::train function allows for cross-validation
using methods like _OOB_ and _repeatedcv_, leading to lower out of sample errors. The selected models could predict the 
'way' of the barbell lift with 100% accuracy on a validation set of 3134 observations. 

Of the two algorithms used, namely _Random Forest_ and _Gradient Boosting_, Random Forests seems
better suited for the current problem, specially since random forest models could be easily built with low
computation power like that available on personal computers.

The model $mrf200$ further reveals that, out of the 159 features present in the provided data,  *yaw_belt*,
*pitch_belt*, *magnet_dumbbell_z*, *magnet_dumbbell_y* and *pitch_forearm* are the top 5 most important
features for predicting how a barbell lift was performed.  

&nbsp;  

&nbsp;  

&nbsp;  

---


[1]: https://www.coursera.org/learn/practical-machine-learning/supplement/PvInj/course-project-instructions-read-first  
[2]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
[3]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
[4]: http://groupware.les.inf.puc-rio.br/har
[5]: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har
[7]: testModels.html
