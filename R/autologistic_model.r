
##ARSENIC DATA ANALYSIS

#install.packages("ggplot2")
#install.packages("sp")
#install.packages("lctools")
#install.packages("spdep")
#install.packages("spatialEco")
library(spatialEco)
library(sp)
library(ggplot2)
library(spdep)
library(ggpubr)
library(caret)
theme_set(theme_pubr())


setwd("F:/github/arsenic-modeling/data")
mydata <- read.csv("Final_Arsenic_DataNew2.csv", sep = ',')
colnames(mydata)


# select only the variables important for the modelling
ArsenicData = subset(mydata, select = c(1,14,27:37))
head(ArsenicData)


# transform the data into factors
ArsenicData$Arsenic_Detect2 = as.factor(ArsenicData$Arsenic_Detect2)
levels(ArsenicData$Arsenic_Detect2) = c('No','Yes')
colnames(ArsenicData)


# transform categorical variables into factors
ArsenicData$Depth = as.factor(ArsenicData$Depth)
ArsenicData$Bedrock = as.factor(ArsenicData$Bedrock)
ArsenicData$BedrockNew = as.factor(ArsenicData$BedrockNew)
attach(ArsenicData)
head(ArsenicData)
colnames(ArsenicData)

# Summary table
summaryvar = summary(ArsenicData)
summaryvar
write.csv(summaryvar, file = "summary.csv")

# Plotting histograms of the variables
pHplot = ggplot(ArsenicData, aes(x=pH))+ geom_histogram(binwidth=1, bins = 14, color="darkblue", fill="skyblue3")+ scale_x_continuous(name="pH Level", breaks = c(0,2, 4, 6, 8, 10, 12,14)) + scale_y_continuous(name="Frequency")+ theme_pubclean()
pHplot
Rocktypeplot = ggplot(ArsenicData, aes(Bedrock)) + geom_bar(color="darkblue", fill="skyblue3") + theme_pubclean() + labs(y="Frequency")
Rocktypeplot
Depthplot = ggplot(ArsenicData, aes(Depth)) + geom_bar(color="darkblue",
fill="skyblue3") + theme_pubclean() + labs(y="Frequency")
Depthplot
colnames(ArsenicData)

# Model Development

# Ordinary logistic regression
lmodel =logistic.regression(ArsenicData, y = 'Arsenic_Detect', x = c('BedrockNew','Depth', 'pH'), penalty = TRUE)
lmodel$model
lmodel$diagTable
lmodel$coefTable
lmodel_pred = predict(lmodel$model, type = 'fitted.ind')

# Spatial autologistic regression
coordinates(ArsenicData) = ~xloc + yloc 
lmodel2 = logistic.regression(ArsenicData, y = 'Arsenic_Detect', x = c('BedrockNew','pH','Depth'), autologistic = TRUE, coords = coordinates(ArsenicData),longlat = FALSE, penalty = TRUE)
lmodel2$model
lmodel2$diagTable
lmodel2$coefTable
lmodel2$bandwidth
lmodel2_pred = predict(lmodel2$model, type = 'fitted.ind')
autocovariate = lmodel2$AutoCov
residuals2 = lmodel2$Residuals


# write results to csv for mapping
model_arsenic = data.frame(DataID,lmodel2_pred, Arsenic_Detect,autocovariate, residuals2, xloc, yloc)
write.csv(model_arsenic, file = "model_arsenic.csv")

# ROC Curve
# install.packages("ROCR")
library(ROCR)
library(pROC)
arsenic_chk = ArsenicData$Arsenic_Detect2

# par(pty ="s")
# roc(arsenic_chk, lmodel_pred, plot = TRUE, legacy.axes = TRUE, xlab = "1-specificity (False positive rate)", ylab = "Sensitivity (True positive rate)", col="#de2d26", lwd=1, print.auc = TRUE)
# plot.roc(arsenic_chk, lmodel2_pred,col="#377eb8", lwd=1, print.auc = TRUE, add=TRUE, print.auc.y=0.4)
# legend("bottomright", legend = c("non-spatial", "spatial"), col = c("#de2d26","#377eb8" ), lwd = 1)

# Area Under the Receiver Operator Characteristic Curve (AUROC)
chkroc1 = pROC::roc(ArsenicData$Arsenic_Detect2,lmodel2_pred)
chkroc1
ci.auc(chkroc1)

# Checking accuracy of predictions
lmodel2_pred0 = rep("No",990)
lmodel2_pred0[lmodel2_pred>.5] ="Yes"
table(lmodel2_pred0, ArsenicData$Arsenic_Detect2)
mean(lmodel2_pred0==ArsenicData$Arsenic_Detect2)
summary(ArsenicData$Arsenic_Detect2)

# Checking the mean squared errors for the testing data set
n0=length(ArsenicData$Arsenic_Detect)
sse10 = sum((ArsenicData$Arsenic_Detect - lmodel2_pred)^2)
mse10 = sse10 / (n0 - 2)
mse10