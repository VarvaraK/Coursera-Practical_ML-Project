### Practical ML : project R code with comments

## set.seed(1984)

## clear workspacespace

rm(list = ls())

## load packages/libraries

library(caret)
library(rpart)
library(lattice)
library(ggplot2)
library(rattle)

## Step 1: Load the data : "/Users/varvarakulikova/Documents/COURSEWORK/Coursera/Practical_ML/Project"

training =  read.csv(file = "pml-training.csv", header = TRUE, na.strings=c("NA","#DIV/0!",""))
testing =  read.csv(file = "pml-testing.csv", header = TRUE, na.strings=c("NA","#DIV/0!",""))

str(training)

summary(training$classe)

summary(training$user_name)

summary(training$cvtd_timestamp)

## Step 2: Partition the data 

inTrain = createDataPartition(y = training$classe, p = 0.6, list = FALSE)
training1 = training[inTrain, ]
testing1 = training[ -inTrain, ]

## Step 3: Cleaning the data

# Identify and throw out variables with near zero variance

NZV = nearZeroVar(training1, freqCut = 90/10, uniqueCut = 20, saveMetrics=TRUE)

head(NZV)
head(NZV$nzv)
rownames(NZV[NZV$nzv ==TRUE,])

NZV_varaibles = names(training1) %in% c(rownames(NZV[NZV$nzv ==TRUE,]))

training1_NZV = training1[!NZV_varaibles]

dim(training1_NZV)

# Identify and put aside variables with more than 70% NA

na_thrhold = 0.9 # threshold for % of NA'S
counter = numeric()
for (i in 1:dim(training1_NZV)[2]){
  if ( sum( is.na(training1_NZV[,i]) )/dim(training1_NZV)[1] >= na_thrhold )
    counter = c(counter, i)
}

training1_NZV_NA = training1_NZV[,-counter]

# Cleaning testing dataset

testing1_NZV_NA = testing1[colnames(training1_NZV_NA)]

# Step 4: Fit Decision Tree model (DT)

modFit_DT = rpart(classe ~ ., data = training1_NZV_NA[,-1], method = "class")
summary(modFit_DT)

library(rattle)
fancyRpartPlot(modFit_DT)

predict_DT = predict(modFit_DT, testing1_NZV_NA[,-1], type = "class")

confusionMatrix(predict_DT, testing1_NZV_NA$classe)

# Predict on the original testing set using DT

dim(testing1)
testing_NZV_NA = testing[,names(testing) %in% names(training1_NZV_NA)]
unique(names(training1_NZV_NA), names(testing))
predict_DT_test = predict(modFit_DT, testing_NZV_NA[,-1], type = "class")

## Fit Random Forest (RF)

modFit_RF = randomForest(classe ~ ., data = training1_NZV_NA[,-1] )

predict_RF = predict(modFit_RF, testing1_NZV_NA[,-1], type = "class")
confusionMatrix(predict_RF, testing1_NZV_NA$classe)

# predict on the original testing set using RF
#To make sure that RandomForest prediction works with the Test data set
#we need to coerce the data into the same type
### This is weird, but it fixes the bug in random.forest.predict function.

for (i in 1:length(testing_NZV_NA) ) {
  for(j in 1:length(training1_NZV_NA)) {
    if( length( grep(names(training1_NZV_NA[i]), names(testing_NZV_NA)[j]) ) ==1)  {
      class(testing_NZV_NA[j]) <- class(training1_NZV_NA[i])
    }      
  }      
}

## corce two data frames, i suspect it passes all the variable types and levels for factors, so they match for bpth data frames
anyrow = 50
testing_NZV_NA = rbind(training1_NZV_NA[anyrow, -56] , testing_NZV_NA) 
testing_NZV_NA = testing_NZV_NA[-1,]

predict_RF_test = predict(modFit_RF, testing_NZV_NA[,-1], type = "class")

predict_DT_test

### Lasso wiith glmnet package

# library(lars)
# library(glmnet)
# 
# y = training1_NZV_NA$classe
# 
# xfactors = model.matrix(training1_NZV_NA$class ~training1_NZV_NA$user_name + training1_NZV_NA$cvtd_timestamp )[,-1]
# head(xfactors)
# 
# x = as.matrix(data.frame(training1_NZV_NA[,setdiff(colnames(training1_NZV_NA),c("user_name", "cvtd_timestamp", "classe") )]))
# 
# head(x)
# modFit_lassoGLM = glmnet(x, y, alpha = 1, family = "multinomial")
# 
# summary(modFit_lassoGLM)
# 
# plot(modFit_lassoGLM, xvar = "lambda")
# grid()
# 
# cv.glmmod = cv.glmnet(x,y = as.numeric(training1_NZV_NA$classe),alpha=1)
# 
# plot(cv.glmmod)
# best_lambda = cv.glmmod$lambda.min
# 
# log(best_lambda )
# 
# 
# x_test = as.matrix(data.frame(testing_NZV_NA[,setdiff(colnames(testing_NZV_NA),c("user_name", "cvtd_timestamp") )]))
# predict_Lasso_test = predict(modFit_lassoGLM, x_test, type = "class")


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predict_RF_test)





