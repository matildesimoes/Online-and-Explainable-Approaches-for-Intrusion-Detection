
library(C50)

train_df <- read.csv("c50_surrogate_results_hst/c50_train.csv", stringsAsFactors=TRUE)
test_df  <- read.csv("c50_surrogate_results_hst/c50_test.csv",  stringsAsFactors=TRUE)

target_name <- "target_surrogate"
train_df[[target_name]] <- as.factor(train_df[[target_name]])
test_df[[target_name]]  <- as.factor(test_df[[target_name]])

for (col in names(train_df)) {
  if (is.factor(train_df[[col]]) && col != target_name) {
    test_df[[col]] <- factor(test_df[[col]], levels=levels(train_df[[col]]))
  }
}

x_train <- subset(train_df, select=-c(target_surrogate))
y_train <- train_df[[target_name]]
x_test  <- subset(test_df,  select=-c(target_surrogate))
y_test  <- test_df[[target_name]]

model_tree  <- C5.0(x_train, y_train, trials=1, rules=FALSE)
model_rules <- C5.0(x_train, y_train, trials=1, rules=TRUE)

pred <- predict(model_tree, x_test)

write.csv(
  data.frame(y_true=y_test, y_pred=pred),
  "c50_surrogate_results_hst/c50_predictions.csv",
  row.names=FALSE
)

capture.output(summary(model_tree),  file="c50_surrogate_results_hst/c50_summary.txt")
capture.output(model_tree,           file="c50_surrogate_results_hst/c50_tree.txt")
capture.output(summary(model_rules), file="c50_surrogate_results_hst/c50_rules.txt")

imp_usage  <- C5imp(model_tree, metric="usage")
imp_splits <- C5imp(model_tree, metric="splits")

imp_df <- data.frame(
  feature = row.names(imp_usage),
  usage   = imp_usage[,1],
  splits  = imp_splits[,1],
  row.names = NULL
)

write.csv(
  imp_df,
  "c50_surrogate_results_hst/c50_variable_importance.csv",
  row.names=FALSE
)

print(model_tree)
cat("\n--- Variable importance (usage) ---\n")
print(imp_df[order(-imp_df$usage),])
