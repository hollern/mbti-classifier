library(tidyverse)
library(caret)

# Read in data
dataset_original <- read.csv('data/mbti_kaggle.csv', stringsAsFactors = FALSE, fileEncoding="latin1", quote="")
mbti_types <- read.csv('data/mbti_types.csv')

# Filter out URLs ||| where string contains 'http://'
dataset_original <- filter(dataset_original, !grepl('http://|https://', post))

# Factor MBTI Type
dataset_original$type <- factor(dataset_original$type, labels = c(0:15),
                                levels = mbti_types$MBTI)

mbti_types_alt <- mbti_types
mbti_types_alt$MBTI <- factor(mbti_types_alt$MBTI, labels = c(0:15), levels = mbti_types$MBTI)
mbti_types_alt$RealType <- factor(mbti_types$MBTI, levels = mbti_types$MBTI)

# Get a sample of 15 of each type
set.seed(100)
indexes <- dataset_original %>%
  group_by(type) %>%
  sample_n(15)
indexes <- indexes$idx
dataset_original <- dataset_original %>%
  filter(idx %in% indexes)
rm(indexes)

# Load into corpus
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$post))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$type = dataset_original$type

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$type, SplitRatio = 0.8)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

rm(corpus, dtm, split)

# === DECISION TREE ===
library(rpart)
library(rpart.plot)
dtree <- rpart(type ~ .,
               data = train,
               method = "class")

rpart.plot(x = dtree, yesno = 2, type = 0, extra = 0)

dtree.pred <- predict(dtree, newdata = test, type = "class")
confusionMatrix(dtree.pred, test$type)

# Plot
for.dtree.graph.2 <- dataset %>%
  mutate(pred = predict(dtree, newdata = dataset, type = "class"))
for.dtree.graph.2$idx <- dataset_original$idx

for.dtree.graph.2 %>%
  inner_join(mbti_types_alt, by = c("pred"="MBTI")) %>%
  group_by(idx, type) %>%
  summarize(avgX = mean(as.numeric(xMBTI)), avgY = mean(as.numeric(yMBTI))) %>%
  inner_join(mutate(mutate(rownames_to_column(mbti_types_alt, "j"), j=as.numeric(j)-1), j=factor(j)), by = c("type"="j")) %>%
  mutate(avgX = avgX - 0.5, avgY = avgY - 0.5) %>%
  ggplot(aes(x = avgX, y = avgY, col = factor(RealType))) +
  geom_text(label='ESFJ', x = 0.875, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ISFJ', x = 1.625, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ESFP', x = 0.875, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISFP', x = 1.625, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISTJ', x = 2.375, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ESTJ', x = 3.125, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ISTP', x = 2.375, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ESTP', x = 3.125, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ENFP', x = 0.875, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='INFP', x = 1.625, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='ENFJ', x = 0.875, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INFJ', x = 1.625, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INTP', x = 2.375, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='ENTP', x = 3.125, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='INTJ', x = 2.375, y = 3.125, size = 5, color = '#aec6cf') +
  geom_text(label='ENTJ', x = 3.125, y = 3.125, size = 5, color = '#aec6cf') +
  geom_point(size = 5, alpha = 0.6) +
  ggtitle('Myers-Briggs Type Predictions', subtitle = 'Predicted Y Value described by Predicted X Value') +
  xlab('Predicted X Values') +
  ylab('Predicted Y Values') +
  scale_x_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  scale_y_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  stat_ellipse(inherit.aes = TRUE) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text = element_blank(),
        axis.ticks = element_blank()) +
  geom_jitter()

# === BAGGED TREE ===
library(ipred)
library(Metrics)
btree <- bagging(type ~ .,
                 data = train,
                 coob = TRUE)

print(btree)

btree.pred <- predict(btree, newdata = test, type = "class")
confusionMatrix(btree.pred, test$type)

# btree.pred.test <- predict(btree, newdata = test, type = "prob")
btree.pred.test <- predict(btree, newdata = test, type = "class")
auc(actual = test$type, predicted = btree.pred.test)

# Plot
for.btree.graph.2 <- dataset %>%
  mutate(pred = predict(btree, newdata = dataset, type = "class"))
for.btree.graph.2$idx <- dataset_original$idx

for.btree.graph.2 %>%
  inner_join(mbti_types_alt, by = c("pred"="MBTI")) %>%
  group_by(idx, type) %>%
  summarize(avgX = mean(as.numeric(xMBTI)), avgY = mean(as.numeric(yMBTI))) %>%
  inner_join(mutate(mutate(rownames_to_column(mbti_types_alt, "j"), j=as.numeric(j)-1), j=factor(j)), by = c("type"="j")) %>%
  mutate(avgX = avgX - 0.5, avgY = avgY - 0.5) %>%
  ggplot(aes(x = avgX, y = avgY, col = factor(RealType))) +
  geom_text(label='ESFJ', x = 0.875, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ISFJ', x = 1.625, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ESFP', x = 0.875, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISFP', x = 1.625, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISTJ', x = 2.375, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ESTJ', x = 3.125, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ISTP', x = 2.375, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ESTP', x = 3.125, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ENFP', x = 0.875, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='INFP', x = 1.625, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='ENFJ', x = 0.875, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INFJ', x = 1.625, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INTP', x = 2.375, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='ENTP', x = 3.125, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='INTJ', x = 2.375, y = 3.125, size = 5, color = '#aec6cf') +
  geom_text(label='ENTJ', x = 3.125, y = 3.125, size = 5, color = '#aec6cf') +
  geom_point(size = 5, alpha = 0.6) +
  ggtitle('Myers-Briggs Type Predictions', subtitle = 'Predicted Y Value described by Predicted X Value') +
  xlab('Predicted X Values') +
  ylab('Predicted Y Values') +
  scale_x_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  scale_y_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  stat_ellipse(inherit.aes = TRUE) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text = element_blank(),
        axis.ticks = element_blank())

# Cross-validate
btree.control <- trainControl(method = "cv",
                              number = 5,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

set.seed(100)
# btree.caret.model <- train(Attrition ~ .,
#                           data = train,
#                           method = "treebag",
#                           metric = "ROC",
#                           trControl = btree.control)
btree.caret.model <- train(type ~ .,
                           data = train,
                           method = "treebag",
                           metric = "Accuracy")

# btree.caret.model.pred <- predict(btree.caret.model,
#                                   newdata = test,
#                                   type = "prob")
btree.caret.model.pred <- predict(btree.caret.model,
                                  newdata = test,
                                  type = "class")

auc(actual = test$type, predicted = btree.caret.model.pred)

# === RANDOM FOREST ===
library(randomForest)
set.seed(100)
rforest <- randomForest(x = train[-1563],
                        y = train$type,
                        ntree = 10)

rforest

plot(rforest)
legend(x = "right", legend = colnames(rforest$err.rate), fill = 1:ncol(rforest$err.rate))

rforest.pred <- predict(rforest, newdata = test, type = "class")
confusionMatrix(rforest.pred, test$type)

# rforest.pred.test <- predict(rforest, newdata = test, type = "pred")
rforest.pred.test <- predict(rforest, newdata = test, type = "class")
auc(actual = test$type, predicted = rforest.pred.test)

# We'll be able to plot sensitivity and specificity by mbti type

# Plot
for.rforest.graph <- dataset_original[row.names(test), ]
for.rforest.graph$pred <- rforest.pred.test

for.rforest.graph %>%
  inner_join(mbti_types_alt, by = c("pred"="MBTI")) %>%
  group_by(idx, type) %>%
  summarize(avgX = mean(as.numeric(xMBTI)), avgY = mean(as.numeric(yMBTI))) %>%
  inner_join(mutate(mutate(rownames_to_column(mbti_types_alt, "j"), j=as.numeric(j)-1), j=factor(j)), by = c("type"="j")) %>%
  ggplot(aes(x = avgX, y = avgY, col = factor(RealType))) +
  geom_text(label='ESFJ', x = 0.875, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ISFJ', x = 1.625, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ESFP', x = 0.875, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISFP', x = 1.625, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISTJ', x = 2.375, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ESTJ', x = 3.125, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ISTP', x = 2.375, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ESTP', x = 3.125, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ENFP', x = 0.875, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='INFP', x = 1.625, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='ENFJ', x = 0.875, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INFJ', x = 1.625, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INTP', x = 2.375, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='ENTP', x = 3.125, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='INTJ', x = 2.375, y = 3.125, size = 5, color = '#aec6cf') +
  geom_text(label='ENTJ', x = 3.125, y = 3.125, size = 5, color = '#aec6cf') +
  geom_point(size = 5, alpha = 0.6) +
  ggtitle('Myers-Briggs Type Predictions', subtitle = 'Predicted Y Value described by Predicted X Value') +
  xlab('Predicted X Values') +
  ylab('Predicted Y Values') +
  scale_x_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  scale_y_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  stat_ellipse(inherit.aes = TRUE) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text = element_blank(),
        axis.ticks = element_blank())

for.rforest.graph.2 <- dataset %>%
  mutate(pred = predict(rforest, newdata = dataset, type = "class"))
for.rforest.graph.2$idx <- dataset_original$idx

for.rforest.graph.2 %>%
  inner_join(mbti_types_alt, by = c("pred"="MBTI")) %>%
  group_by(idx, type) %>%
  summarize(avgX = mean(as.numeric(xMBTI)), avgY = mean(as.numeric(yMBTI))) %>%
  inner_join(mutate(mutate(rownames_to_column(mbti_types_alt, "j"), j=as.numeric(j)-1), j=factor(j)), by = c("type"="j")) %>%
  mutate(avgX = avgX - 0.5, avgY = avgY - 0.5) %>%
  ggplot(aes(x = avgX, y = avgY, col = factor(RealType))) +
  geom_text(label='ESFJ', x = 0.875, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ISFJ', x = 1.625, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ESFP', x = 0.875, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISFP', x = 1.625, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISTJ', x = 2.375, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ESTJ', x = 3.125, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ISTP', x = 2.375, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ESTP', x = 3.125, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ENFP', x = 0.875, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='INFP', x = 1.625, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='ENFJ', x = 0.875, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INFJ', x = 1.625, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INTP', x = 2.375, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='ENTP', x = 3.125, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='INTJ', x = 2.375, y = 3.125, size = 5, color = '#aec6cf') +
  geom_text(label='ENTJ', x = 3.125, y = 3.125, size = 5, color = '#aec6cf') +
  geom_point(size = 5, alpha = 0.6) +
  ggtitle('Myers-Briggs Type Predictions', subtitle = 'Predicted Y Value described by Predicted X Value') +
  xlab('Predicted X Values') +
  ylab('Predicted Y Values') +
  scale_x_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  scale_y_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  stat_ellipse(inherit.aes = TRUE) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text = element_blank(),
        axis.ticks = element_blank())

# Tune
set.seed(100)
rforest.res <- tuneRF(x = subset(train, select = -type),
                      y = train$type, ntreeTry = 500)
print(rforest.res)
rforest.mTryOpt <- rforest.res[, "mtry"][which.min(
  rforest.res[, "OOBError"])]
print(rforest.mTryOpt)

# List of possible values mtry, nodesize and sampsize
rforest.mtry <- seq(4, ncol(train) * 0.8, 2)
rforest.nodesize <- seq(3, 8, 2)
rforest.sampsize <- nrow(train) * c(0.7, 0.8)

rforest.hyperGrid <- expand.grid(mtry = rforest.mtry,
                                 nodesize = rforest.nodesize,
                                 sampsize = rforest.sampsize)
rforest.oobErr <- c()

# === BOOSTING ===
library(gbm)
set.seed(100)
gboost <- gbm(type ~ .,
              distribution = "multinomial",
              data = train,
              n.trees = 10000)
print(gboost)
# summary(gboost)

gboost.pred.norm <- predict(gboost, newdata = test, n.trees = 10000)
gboost.pred.resp <- predict(gboost, newdata = test, n.trees = 10000, type = "response")

# Plot
for.gboost.graph.2 <- dataset %>%
  mutate(pred = predict(gboost, newdata = dataset, n.trees = 500, type = "response"))
for.gboost.graph.2$idx <- dataset_original$idx

for.gboost.graph.2 %>%
  inner_join(mbti_types_alt, by = c("pred"="MBTI")) %>%
  group_by(idx, type) %>%
  summarize(avgX = mean(as.numeric(xMBTI)), avgY = mean(as.numeric(yMBTI))) %>%
  inner_join(mutate(mutate(rownames_to_column(mbti_types_alt, "j"), j=as.numeric(j)-1), j=factor(j)), by = c("type"="j")) %>%
  mutate(avgX = avgX - 0.5, avgY = avgY - 0.5) %>%
  ggplot(aes(x = avgX, y = avgY, col = factor(RealType))) +
  geom_text(label='ESFJ', x = 0.875, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ISFJ', x = 1.625, y = 0.875, size = 5, color = '#fdfd96') +
  geom_text(label='ESFP', x = 0.875, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISFP', x = 1.625, y = 1.625, size = 5, color = '#fdfd96') +
  geom_text(label='ISTJ', x = 2.375, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ESTJ', x = 3.125, y = 0.875, size = 5, color = '#ffb347') +
  geom_text(label='ISTP', x = 2.375, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ESTP', x = 3.125, y = 1.625, size = 5, color = '#ffb347') +
  geom_text(label='ENFP', x = 0.875, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='INFP', x = 1.625, y = 2.375, size = 5, color = '#77dd77') +
  geom_text(label='ENFJ', x = 0.875, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INFJ', x = 1.625, y = 3.125, size = 5, color = '#77dd77') +
  geom_text(label='INTP', x = 2.375, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='ENTP', x = 3.125, y = 2.375, size = 5, color = '#aec6cf') +
  geom_text(label='INTJ', x = 2.375, y = 3.125, size = 5, color = '#aec6cf') +
  geom_text(label='ENTJ', x = 3.125, y = 3.125, size = 5, color = '#aec6cf') +
  geom_point(size = 5, alpha = 0.6) +
  ggtitle('Myers-Briggs Type Predictions', subtitle = 'Predicted Y Value described by Predicted X Value') +
  xlab('Predicted X Values') +
  ylab('Predicted Y Values') +
  scale_x_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  scale_y_continuous(limits = c(0.5,3.5), breaks = c(0.5, 2.0, 3.5), minor_breaks = c(1.25, 2.0, 2.75)) +
  stat_ellipse(inherit.aes = TRUE) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text = element_blank(),
        axis.ticks = element_blank())

paste(range(gboost.pred.norm), range(gboost.pred.resp), sep = " ")

auc(actual = test$type, predicted = gboost.pred.norm)
auc(actual = test$type, predicted = gboost.pred.resp)

# Hyperparameters to reduce overfitting
gboost.ntree.opt.oob <- gbm.perf(object = gboost,
                                 method = "OOB",
                                 oobag.curve = TRUE)

set.seed(100)
gboost.cv <- gbm(type ~ .,
                 distribution = "gaussian",
                 data = train,
                 n.trees = 10000,
                 cv.folds = 2)

gboost.ntree.opt.cv <- gbm.perf(object = gboost.cv, method = "cv")

print(paste0("Optimal n.trees (OOB): ", gboost.ntree.opt.oob))
print(paste0("Optimal n.trees (CV): ", gboost.ntree.opt.cv))

gboost.pred.oob <- predict(gboost, newdata = test, n.trees = gboost.ntree.opt.oob)
gboost.pred.cv <- predict(gboost.cv, newdata = test, n.trees = gboost.ntree.opt.cv)

gboost.auc.oob <- auc(actual = test$type,
                      predicted = gboost.pred.oob)
gboost.auc.cv <- auc(actual = test$type,
                     predicted = gboost.pred.cv)

paste0("Test set AUC (OOB): ", gboost.auc.oob)
paste0("Test set AUC (CV): ", gboost.auc.cv)



# === COMPARE THE MODELS ===
actual <- test$type
dt_auc <- auc(actual = actual, predicted = dtree.pred)
bt_auc <- auc(actual = actual, predicted = btree.pred)
rf_auc <- auc(actual = actual, predicted = rforest.pred)
gb_auc <- auc(actual = actual, predicted = gboost.pred.cv)

sprintf("Decision Tree Test AUC: %.3f", dt_auc)
sprintf("Bagged Trees Test AUC: %.3f", bt_auc)
sprintf("Random Forest Test AUC: %.3f", rf_auc)
sprintf("GBM Test AUC: %.3f", gb_auc)

preds <- list(dtree.pred, btree.pred, rforest.pred, gboost.pred.cv)

m <- length(plist)
actuals <- rep(list(test$Attrition), m)

# Plot ROC Curves
library(ROCR)
pred <- prediction(preds, actuals)
rocs <- performance(pred, "tpr", "fpr")

plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves [TP by FP]",
     xlab = rocs@x.name, ylab = rocs@y.name)
legend(x = "bottomright",
       legend = c("Decision Tree", "Bagged Tree", "Random Forest", "GBM"),
       fill = 1:m)