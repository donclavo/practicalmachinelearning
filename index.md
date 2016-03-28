# Predicting Weigth Lifting Performance 
Javier Clavijo  
March 27th - 2016  

## Overview
In this document I will analyze the Weight Lifting Exercise Dataset, which contains information of 4 different sensors in subjects that are performing Weight Lifting activities. In particular, this document presents a methodology for predicting how well the subject is performing the exercise, in accordance with a guided classification in 5 different categories that helps training the Machine Learning Algorithms that performs the task. At the end, I will show the performance of the most successful methodologies for this dataset, which were Random Trees and Boosting with trees algorithms. 


## Exploring and cleaning the Data


```r
Data<-read.csv("pml-training.csv")
library(caret)
dim(Data)
```

```
## [1] 19622   160
```
After reading the training Data, we realized that this is a relative big data set, having a great amount of columns and Data to process.

In the first place, a simple check of the data shows some insights:

1. The first columns register some design parameters that should not be relevant for predicting the quality of the exercise (with the exception of "user name").
2. For each place of measure in the body, there are some information columns, and some summary columns. 
3. There are some columns that have a great amount of NA values, and others with many empty values. It seems to be a relation between the column "New window" and the erroneous values, as the erroneous values occur in the summary columns, for the registers with "New Window" equal to "yes". Checking the test set, I could realize that it doesn't have any of these type of registers.

The next lines of code cleans the registers with "new Window = yes", the summary columns (with erroneous data) and the columns that only contain information of the design of the experiment.


```r
CleanData<-Data[Data$new_window=="no",]

CleanData<-CleanData[,-grep("kurt",names(CleanData))]
CleanData<-CleanData[,-grep("skew",names(CleanData))]
CleanData<-CleanData[,-grep("min",names(CleanData))]
CleanData<-CleanData[,-grep("max",names(CleanData))]
CleanData<-CleanData[,-grep("amplit",names(CleanData))]
CleanData<-CleanData[,-grep("var",names(CleanData))]
CleanData<-CleanData[,-grep("avg",names(CleanData))]
CleanData<-CleanData[,-grep("std",names(CleanData))]

CleanData<-CleanData[,-c(1,3:7)]
```

The remaining dataset contains the features that are going to be used for the rest of the document.

## Error design
For accepting a prediction algorithm, it will be expected that the out of sample error will be less than 20% (Accuracy larger than 80%).

## Training the model

### Classification Tree
The first approach I took was to build a simple classification tree. For this purpose, I first perform cross validation, dividing the training data in two different sets, a training set with 75% of data and a test set with the remaining 25%.

```r
set.seed(6532)
inTrain = createDataPartition(CleanData$classe, p = 3/4)[[1]]
training = CleanData[ inTrain,]
testing = CleanData[-inTrain,]
```

Using the caret package, I train the model as follows.


```r
set.seed(952)
ModTree<-train(classe~.,method="rpart",data=training)
```

The result can be seen in the following figure.

###Figure 1: Classification tree

```r
plot(ModTree$finalModel,uniform = TRUE,main="Classification tree for WLE")
text(ModTree$finalModel,use.n = TRUE,all=TRUE,cex=0.6)
```

![](index_files/figure-html/Plotting Classification Tree-1.png)

The plot shows a simple classification tree. Given that each branch is created by only one rule concerning a single column, but that the Weight Lifting exercise is measured by four different instruments, it is expected that this algorithm does not have the best performance, as a single result hardly will be sufficient for predicting the outcome of the exercise. Below I show the results of the prediction in the testing set.


```r
predTree<-predict(ModTree,testing)
Treeaccuracy<-sum(predTree==testing$classe)/nrow(testing)
print(Treeaccuracy)
```

```
## [1] 0.4981258
```

As shown, this prediction algorithm only classifies correctly around half of the testing samples. In order to get better predictions, it is required to use more complicated models.

### Cross Validation

Given the great amount of data of the Dataset, and the machine processing limitations, for the next more complex models I must change the cross validation strategy. For the rest of the document, I will use only the 10% of the data as the training set, and I will divide the testing set in two parts (testing and validation), as follows.


```r
set.seed(3181)
inTrain = createDataPartition(CleanData$classe, p = 1/10)[[1]]
training = CleanData[ inTrain,]
temp<-CleanData[-inTrain,]
inTrain = createDataPartition(temp$classe, p = 1/2)[[1]]
testing = temp[inTrain,]
validation = temp[-inTrain,]
```

With this strategy, it will be possible to combine the most successful predictors to check if with a combined predictor we can obtain better results.

Now, I will perform three different algorithms.

### Random Forest

Using the training data, the random forest predictor is trained.


```r
set.seed(546)
ModRF<-train(classe~.,method="rf",data=training)
```
The prediction is evaluated against the testing data set.


```r
predRF<-predict(ModRF,testing)
RFaccuracy<-sum(predRF==testing$classe)/nrow(testing)
print(RFaccuracy)
```

```
## [1] 0.9576732
```

The algorithm performs very well, obtaining an over 95% accuracy on the testing set. This implies that it is not necessary to compute a larger training set, which would imply the utilization of methods as Principal Components Analysis.

### Boosting

Analogously, I train the Boosting with trees algorithm, and observe the prediction performance.


```r
set.seed(2938)
ModGBM<-train(classe~.,method="gbm",data=training,verbose=FALSE)
```



```r
predGBM<-predict(ModGBM,testing)
GBMaccuracy<-sum(predGBM==testing$classe)/nrow(testing)
print(GBMaccuracy)
```

```
## [1] 0.9277206
```
This algorithm also has a very good performance (over 90% accuracy). 

### Linear Discriminant Analysis

Finally, the same procedure is performed with this last method.


```r
set.seed(547)
ModLDA<-train(classe~.,method="lda",data=training)
```



```r
predLDA<-predict(ModLDA,testing)
LDAaccuracy<-sum(predLDA==testing$classe)/nrow(testing)
print(LDAaccuracy)
```

```
## [1] 0.718862
```
The performance of this algorithm is still better than simple classification trees, but it presents an important error (larger than 25%).

## Combining predictors
As a final step, I combine the two predictors with the most accuracy (Random forest and Boosting), to test if the two predictors together can perform even better.

First, I use an in sample error algorithm, to check if it is surpassing the previous accuracy.


```r
set.seed(3256)
predDF<-data.frame(predRF,predGBM,classe=testing$classe)
ModComb<-train(classe~.,method="rf",data=predDF)
predComb<-predict(ModComb,predDF)
Combaccuracy<-sum(predComb==testing$classe)/nrow(testing)
print(Combaccuracy)
```

```
## [1] 0.9584827
```
As we can see, the error now is slightly smaller than before.

Now, I perform the combination algorithm for the validation set.

```r
set.seed(5214)
predRF_V<-predict(ModRF,validation)
predGBM_V<-predict(ModRF,validation)
predDF_V<-data.frame(predRF_V,predGBM_V,classe=validation$classe)
ModComb_V<-train(classe~.,method="rf",data=predDF_V)
predComb_V<-predict(ModComb_V,predDF_V)

RFaccuracy_V<-sum(predRF_V==validation$classe)/nrow(validation)
print(RFaccuracy_V)
```

```
## [1] 0.9593985
```

```r
GBMaccuracy_V<-sum(predGBM_V==validation$classe)/nrow(validation)
print(GBMaccuracy_V)
```

```
## [1] 0.9595142
```

```r
Combaccuracy_V<-sum(predComb_V==validation$classe)/nrow(validation)
print(Combaccuracy_V)
```

```
## [1] 0.9595142
```
In this case, the prediction was also a little better than before when compared against Random Forest, but equal to Boosting.

The total confussion matrix is presented below.

###Figure 2: Confussion Matrix

```r
confusionMatrix(validation$classe,predComb_V)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2426   16    8    9    2
##          B   63 1552   49    0    9
##          C    0   50 1430   26    2
##          D    2    2   54 1354    4
##          E    1    8   14   31 1533
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9595          
##                  95% CI : (0.9551, 0.9636)
##     No Information Rate : 0.2883          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9488          
##  Mcnemar's Test P-Value : 1.661e-13       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9735   0.9533   0.9196   0.9535   0.9890
## Specificity            0.9943   0.9828   0.9890   0.9914   0.9924
## Pos Pred Value         0.9858   0.9277   0.9483   0.9562   0.9660
## Neg Pred Value         0.9893   0.9891   0.9825   0.9909   0.9976
## Prevalence             0.2883   0.1883   0.1799   0.1643   0.1793
## Detection Rate         0.2806   0.1795   0.1654   0.1566   0.1773
## Detection Prevalence   0.2847   0.1935   0.1744   0.1638   0.1836
## Balanced Accuracy      0.9839   0.9680   0.9543   0.9725   0.9907
```

## Conclussions

The model could be predicted with high accuracy (more than 95%) using only a small portion of the Data to train the algorithms, avoiding simplifying the covariates using methods as Principal Component Analysis. The algorithms with best performance were Random Forest and Boosting with trees. In addition, the combination of predictors for the most successful methods allows us to obtain even better results, decreasing the out of sample error in the validation set. However, the increase in accuracy in this case might be not sufficient for deciding to use this model when reducing computation time and simplicity in the model are required.

