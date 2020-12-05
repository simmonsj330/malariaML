# RANDOM FOREST PIPELINE

library(caret)
library(ranger)
library(tidyverse)
library(doParallel)

##TODO
# 3. Test run
# 5. Update README, note that RF has less plotting built-in

samples<- "example/samples.txt"
expr<- "example/expr.txt"
outname<- "rf"

#####
# LOAD AND ORGANIZE DATA
#####
# Create output directory
dir.create(paste0("./", outname), showWarnings = FALSE)

# Load data
expr<- read.table(expr, header=T, sep="\t")
samples<- read.table(samples, header=T, sep="\t")

# Combine
modelDat<- left_join(samples, expr, by="Genotype_ID")

#####
# MODELLING PREP
#####

# Build training and validation sets
modelDat <- modelDat %>%
  as_tibble() %>%
  group_by(Pheno)

# Shuffle the order of samples
modelDat <- modelDat[sample(nrow(modelDat)),]

# See how many of each class we have
trainSize<- round(0.9*min(tally(modelDat)$n))

# Select training and validation
train<- sample_n(modelDat, trainSize)
test<- modelDat[which(!(modelDat$Genotype_ID %in% train$Genotype_ID)),]

# Drop Sample names for model fitting
train_noSample<- select(train, -Genotype_ID)
test_noSample<- select(test, -Genotype_ID)

# Control for parallel fitting
set.seed(42)
ctrl <- trainControl(method = "repeatedcv",
                     number = 3,
                     repeats = 10,
                     classProbs = T,
                     allowParallel = TRUE)

# Set up parallel params
no_cores <- detectCores() - 2 
registerDoParallel(cores=no_cores) 
#cl <- makeCluster(no_cores)

#####
# MODEL FITTING
#####

# Train model in parallel
set.seed(42)
rfFit <- train(as.factor(Pheno) ~ .,
               data = train_noSample, 
               trControl = ctrl,
               method = "ranger",
               importance = 'impurity')

# Stop core clusters
#stopCluster(cl)

#####
# RESULTS AND SAVE
#####

# Predict on test set
pheno_pred<- predict(rfFit, test_noSample)

# Check results
conf<- confusionMatrix(pheno_pred, as.factor(test_noSample$Pheno))

# Get variable importance
features<-varImp(rfFit)$importance
features$Transcript<- rownames(features)
features<- arrange(features, desc(Overall))
colnames(features)[1]<- "Importance"

# Save selected features
write.table(features, paste0(outname, "/",outname ,"_transcriptImportance.txt"),
            quote = F, sep='\t', col.names = T, row.names = F)

# Save accuracy stats
acc<- data.frame(Lower= conf$overall[3], Mean= conf$overall[1], Upper= conf$overall[4])
write.table(acc, file= paste0(outname, "/",outname, "_accuracy.txt"),
            quote = F, sep='\t', col.names = T, row.names = F)




