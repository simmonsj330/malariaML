# LASSO PIPELINE

library(tidyverse)
library(useful)
library(ggfortify)
library(mgsub)
library(ggplot2)
library("glmnet")
library(lmtest)
library(textclean)
library(ROCR)
# Set my favorite ggplot settings
cai_theme<- theme_gray(base_size = 18) + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5), 
        plot.subtitle= element_text(hjust=0.5), 
        legend.text = element_text(size=12), 
        legend.title = element_text(size = 12))
theme_set(cai_theme)

samples<- "example/samples.txt"
expr<- "example/expr.txt"
outname<- "test"
subtit<- "Test"

#####
# LOAD DATA
#####
# Create output directory
dir.create(paste0("./", outname), showWarnings = FALSE)

# Load data
expr<- read.table(expr, header=T, sep="\t")
samples<- read.table(samples, header=T, sep="\t")

# Define start
expr_start<- ncol(samples)+1

# Reduce expression set to sample set
dat<- inner_join(samples, expr, by="Genotype_ID")
dat$Pheno<- as.factor(dat$Pheno)

#####
# LASSO MODELLING
#####

# Create structures to hold results
model_transcripts<- data.frame(
  transcriptName=c(NA),
  coefficientSum= c(NA),
  modelCount= c(NA)
)

accuracy_distri<- c(NA)

# Iterate 100 times to bootstrap stats
for (count in 1:100) {
  
  # Tracking process
  cat(paste0("Modelling iteration ", count, "\n"))
  
  # Remove NA row and relabel
  if(count==2){
    model_transcripts<- model_transcripts[-c(1),]
    rownames(model_transcripts)<- seq(1, nrow(model_transcripts))
  }
  
  # Build training and validation sets
  model_dat <- dat %>%
    as_tibble() %>%
    group_by(Pheno)
  
  # Shuffle the order of samples
  model_dat <- model_dat[sample(nrow(model_dat)),]
  
  # See how many of each class we have
  trainSize<- round(0.9*min(tally(model_dat)$n))
  
  # Select training and validation
  train<- sample_n(model_dat, trainSize)
  test<- model_dat[which(!(model_dat$Genotype_ID %in% train$Genotype_ID)),]
  
  # Isolate predictors and responses
  train_resp<- as.matrix(train[,c("Pheno")])
  train_pred<- as.matrix(train[,expr_start:ncol(train)])
  test_resp<- as.matrix(test[,c("Pheno")])
  test_pred<- as.matrix(test[,expr_start:ncol(test)])
  
  # Determine optimal lambda value
  fit<- cv.glmnet(train_pred, train_resp, family = "binomial", 
                  type.measure= "class", nfolds = 10)
  
  # Find corresponding minimum misclassification
  mse.min <- fit$cvm[fit$lambda == fit$lambda.min]
  
  # View coefficients
  myCoefs<- coef(fit, s = "lambda.min")
  
  # Assemble into df
  features <- data.frame(
    transcriptName = myCoefs@Dimnames[[1]][which(myCoefs != 0 )],
    coefficientSum    = abs(myCoefs[ which(myCoefs != 0 ) ])
  )
  
  # Drop intercept
  features<- features[-c(1),]

  # Relabel rows
  rownames(features)<- seq(1, nrow(features))
  
  # Predict classes for held-out data
  model_pred<- predict(fit, newx = test_pred, s = "lambda.min", type="class")
  
  # Compute accuracy and append
  accuracy_distri<- append(accuracy_distri, length(which(model_pred==test_resp))/length(model_pred))
  
  # Split features into repeated and new groups
  repeatedRows<- which(features$transcriptName %in% model_transcripts$transcriptName)
  newRows<- which(!(features$transcriptName %in% model_transcripts$transcriptName))
  
  if(length(repeatedRows) > 0){
    
    # Isolate repeats from features
    repeatedTranscripts<- features[repeatedRows,]
    
    # Now find which rows repeats correspond to in model_transcripts
    repeatedRowsinModel<- which(model_transcripts$transcriptName %in% features$transcriptName)

    # Update repeated variables
    model_transcripts[repeatedRowsinModel, 2]<- model_transcripts$coefficientSum[repeatedRowsinModel] + repeatedTranscripts$coefficientSum
    model_transcripts[repeatedRowsinModel, 3]<- model_transcripts[repeatedRowsinModel, 3] + 1
  }
  
  if(length(newRows) > 0){
  
    # Isolate new transcripts
    newTranscripts<- features[newRows,]
    
    # Add new variables to model data
    newTranscripts$modelCount<- 1
    model_transcripts<- rbind(model_transcripts, newTranscripts)
  }
}

# Remove NA row
accuracy_distri<- accuracy_distri[-c(1)]

# Compute average coefficient
model_transcripts<- model_transcripts %>%
  mutate(
    avgCoef= coefficientSum/modelCount
  )

# Sort stats
model_transcripts<- arrange(model_transcripts, desc(modelCount))

# Save lambda optimization plot
png(paste0(outname,"/",outname,"_optimization.png"))
plot(fit)
dev.off() 

# Save list of selected features
write.table(model_transcripts, file =paste0(outname,"/", outname, "_lasso_models.txt") , 
            sep="\t", row.names=FALSE, quote = FALSE)

# Save accuracy stats
write.table(accuracy_distri, file =paste0(outname,"/", outname, "_accuracyDistribution.txt") , 
            sep="\t", row.names=FALSE, quote = FALSE)

#####
# PCA
#####
# Isolate expression data and compute
pca_results <- dat[,expr_start:ncol(dat)] %>%
  prcomp(scale=F, center= T, tol= 0.01)

# Visualize PCA
autoplot(pca_results, data= dat, label= F, colour= "Pheno") +
  labs( title= "Leaf Expression PCA", subtitle = subtit, colour="Pheno") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5), plot.subtitle= element_text(hjust=0.5), 
        legend.text = element_text(size=12), legend.title = element_text(size = 12)) +
  ggsave(paste0(outname,"/",outname,"_pca.png"))


#####
# POST-LASSO PCA
#####

# Select transcript subset
if(nrow(model_transcripts)>20){
  # Reduce transcripts to those selected from prediction
  reduced_trans<- as.data.frame(dat[,which(colnames(dat) %in% model_transcripts$transcriptName[1:20])])
} else{
  reduced_trans<- as.data.frame(dat[,which(colnames(dat) %in% model_transcripts$transcriptName)])
  colnames(reduced_trans)<- c(as.character(model_transcripts[1,1]))
}

# If only one left do expression plot
if(ncol(reduced_trans)==1){
  
  # Combine pca with traits
  plotDat<- dat[,c(1, 3)]
  plotDat<- cbind(plotDat, reduced_trans)
  #colnames(plotDat)[3]<- as.character(model_transcripts[1,1])
  
  # Visualize single dimensional PCA
  ggplot(data= plotDat, aes(seq_along(Genotype_ID), plotDat[,3] , colour= Pheno))+
    geom_point() +
    labs(x= "Samples", y= colnames(plotDat)[3] , title= "Leaf Expression", colour="Pheno",
         subtitle = paste0(subtit, ": LASSO Selected")) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5), legend.text = element_text(size=12),
          legend.title = element_text(size = 12), plot.subtitle = element_text(hjust = 0.5)) +
    ggsave(paste0(outname,"/",outname,"_pca_lassoSelected.png"))

# Else do PCA 
} else{
  
  # Repeat with reduced set
  pca_results <- reduced_trans %>%
    prcomp(scale=F, center= T, tol= 0.01)
  
  # Plot and save
  autoplot(pca_results, data= dat, label= F, colour= "Pheno") +
    labs(colour="Pheno", title= "Leaf Expression PCA", subtitle = paste0(subtit, ": LASSO Selected")) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5), plot.subtitle= element_text(hjust=0.5), 
          legend.text = element_text(size=12), legend.title = element_text(size = 12))
    ggsave(paste0(outname,"/",outname,"_pca_lassoSelected.png"))
}

