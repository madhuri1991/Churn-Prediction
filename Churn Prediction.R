"The churn dataset contains data on a variety of telecom customers and the modeling challenge is 
to predict which customers will cancel their service (or churn)"

library(caret)
library(C50)
library(glmnet)
library(pROC)
library(randomForest)
library(caretEnsemble)

data(churn)
table(churnTrain$churn) / nrow(churnTrain) 

churn_x <- churnTrain[,-20]
churn_y <- churnTrain[,20]

length(churn_y)
dim(churn_x)

# Create custom indices: myFolds
myFolds <- createFolds(churn_y, k = 5)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

# Fit glmnet model: model_glmnet
model_glmnet <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "glmnet",
  trControl = myControl
)


# Fit random forest: model_rf
model_rf <- randomForest(churn ~ ., data=churnTrain, metric = "ROC", trControl = myControl)
summary(model_rf)

# Fit random forest: model_rf
model_rf <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "ranger",
  trControl = myControl
)

# Create model_list
model_list <- list(item1 = model_glmnet, item2 = model_rf)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)

# Create bwplot
# bwplot(resamples)
bwplot(resamples, metric = "ROC")
# Create xyplot
xyplot(resamples, metric="ROC")

# Create ensemble model: stack
stack <- caretStack(model_list, method="glm")

# Look at summary
summary(stack)
















