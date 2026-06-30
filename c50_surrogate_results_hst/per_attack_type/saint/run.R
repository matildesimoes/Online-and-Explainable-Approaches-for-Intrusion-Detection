
library(C50)

tr <- read.csv("c50_surrogate_results_hst/per_attack_type/saint/train.csv", stringsAsFactors=TRUE)
te <- read.csv("c50_surrogate_results_hst/per_attack_type/saint/test.csv", stringsAsFactors=TRUE)

tr$target_surrogate <- as.factor(tr$target_surrogate)
te$target_surrogate <- as.factor(te$target_surrogate)

for (col in names(tr)) {
  if (is.factor(tr[[col]]) && col != "target_surrogate") {
    te[[col]] <- factor(te[[col]], levels=levels(tr[[col]]))
  }
}

mod <- C5.0(subset(tr, select=-c(target_surrogate)), tr$target_surrogate, trials=1)
imp <- C5imp(mod, metric="usage")
imp_df <- data.frame(feature=row.names(imp), usage=imp[,1], row.names=NULL)
write.csv(imp_df, "c50_surrogate_results_hst/per_attack_type/saint/importance.csv", row.names=FALSE)
