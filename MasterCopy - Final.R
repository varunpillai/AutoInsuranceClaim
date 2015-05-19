# GROUP 7 - Predicting Auto Insurance Loss
# 
# Objective: 
# The objective of this project is to analyze historical insurance data to find predictive variables that will ultimately predict an insurance company's future losses.

# Data dictionary and source

# Variable Description
# ----------------------
# Age | Age of policy holder
# Age.Band | Age groups of 16-25, 26-59, and 60+
# Years of Driving Experience | Years driven 
# Number of Vehicles  | Number of Cars insured under policy 
# Gender | M/F
# Married | Married/Single
# Vehicle Age | Age of vehicle insured under policy
# Vehicle Age Band | Groups of Vehicle Age
# Fuel Type | Petro/Diesel
# Losses | Loss claimed under policy
# 
# The data come from an auto insurance company in India, where variables and the background are provided.
# 
# Dataset summary

# LOAD LIBRARY
install.packages("lmtest")
install.packages("sandwich")
install.packages("plyr")
install.packages("ggplot2")
library("lmtest")
library("sandwich")
library("plyr")
library("ggplot2")

# READ THE DATASET
```{r}
rm(list=ls())
InsuranceData=read.csv(file="Insurance Data Updated.csv")
```

```{r}
summary(InsuranceData)
```

# Sampling (Split Dataset Randomly)
### Next we sample 80% of the original data and use it as the training set. The remaining 20% is used as test set. The regression model will be built on the training set and future performance of your model will be evaluated with the test set.

```{r}
set.seed(1000)
subset <- sample(nrow(InsuranceData),nrow(InsuranceData)*0.80)
Insurance_train = InsuranceData[subset,]
Insurance_test = InsuranceData[-subset,]
attach(Insurance_train)
```

# Response Variable Exploration

# Plot of Losses
```{r}
plot(Losses,col=rgb(74,126,187,max=255),pch=18,ylab="Losses",xlab="No. of Records")
grid(nx=NA,ny=NULL,col=rgb(165,165,165,max=255),lty=1)
```

# Histogram of Losses
```{r}
hist(Losses,breaks=100, xlab="Losses (00s)",ylab="Density of Policy", main="Distribution of Losses",col=rgb(218,238,243,max=255),border=0,prob=TRUE)
curve(dnorm(x,mean=mean(Losses),sd=sd(Losses)),col="red",lwd=2,add=TRUE)
lines(density(Losses,adjust=3),col=rgb(112,48,160,max=255),lwd=2)
```

```{r}
summary(Losses)
quantile(Losses,c(0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975,0.99,0.995))
```
fivenum(Losses)
### From the plot graph, we can see quite a few outliners. From the histogram, the losses are highly skewed to the right.  Therefore, we decided to normalize the data. 

# Histogram of LogLosses
```{r}
logloss<-log(Losses)
hist(logloss,breaks=100, xlab="LogLoss",ylab="Density of Policy", main="Distribution of LogLoss",col=rgb(218,238,243,max=255),border=0,prob=TRUE)
curve(dnorm(x,mean=mean(logloss),sd=sd(logloss)),col="red",lwd=2,add=TRUE)
lines(density(logloss,adjust=3),col=rgb(112,48,160,max=255),lwd=2)
```

# Independent Variables Analysis
# Bivariate analysis of response agains independent variables
```{r}
par(mfrow=c (2, 2)) 
plot(logloss ~Age.Band,col="#00000033")
plot(logloss ~Gender,col="#00000033")
plot(logloss ~Married,col="#00000033")
plot(logloss ~Fuel.Type,col="#00000033")
par(mfrow=c (1, 1)) 
plot(logloss ~Vehicle.Age.Band,col="#00000033")
```

# Age group and Gender
```{r}
bar<-ggplot(Insurance_train, aes(Gender,logloss,fill=Age.Band))
bar+ stat_summary(fun.y=mean,geom="histogram", position="dodge", width=0.2)
```

# Age group and Marital status
```{r}
bar<-ggplot(Insurance_train, aes(Married, logloss,fill=Age.Band))
bar+ stat_summary(fun.y=mean,geom="histogram", position="dodge", width=0.2)
```

# Find correlations between variables . Age and Driving experience has 99% correlation.
```{r}
cor(Years.of.Driving.Experience,Age)
cor(logloss, Age)
cor(logloss,Years.of.Driving.Experience)
cor(logloss,Number.of.Vehicles)
```

# Anova Test(for unequal variances ) and T-tests
# Comparing logloss between different age groups
```{r}
oneway.test(logloss ~Age.Band,var.equal=F)
```

# Comparing logloss between gender
```{r}
t.test(logloss ~Gender)
```

# Comparing logloss between marrital status
```{r}
t.test(logloss ~Married)
```

# Comparing logloss between fuel type
```{r}
t.test(logloss ~Fuel.Type)
```


# Model Building and Variable Selection
---------------------------------------
# Initial Model
# The following model includes all $x$ varables in the model
```{r}
model_1 <- lm(logloss ~Number.of.Vehicles+Age.Band:Years.of.Driving.Experience+Gender +Married +Vehicle.Age +Fuel.Type)
summary(model_1)
step(model_1,direction="backward")
```

### Based on the results, the second model has higher AIC which implies better fitness. We then continue to evaluate it based on the value of adjusted R square. 
```{r}
model_2 <- lm(logloss ~Age.Band:Years.of.Driving.Experience+Gender +Married +Vehicle.Age +Fuel.Type)
summary(model_2)
```

# In-Sample Model Evaluation

# MSE comparison
```{r}
modelone_summary <- summary(model_1)
summary(model_1)
mean((modelone_summary$residuals)^2)
```

```{r}
modeltwo_summary <- summary(model_2)
mean((modeltwo_summary$residuals)^2)
```


# R Square comparison
# $R^2$ of the model 1
```{r}
modelone_summary$adj.r.squared
```

# $R^2$ of the model 2
```{r}
modeltwo_summary$adj.r.squared
```

# AIC and BIC comparison

```{r}
AIC(model_1)
AIC(model_2)
```

```{r}
BIC(model_1)
BIC(model_2)
```

# Based on the result, we confirmed that model 2 is our best model.

# Out-of-Sample Prediction
### To evaluate how the model performs on future data, we use predict() to get the predicted values from the test set.
```{r, eval=FALSE}
testdata <- predict(model_2, Insurance_test)
testdata
```
  
# Mean Squared Error (MSE) Comparison
```{r}
tlogloss<-log(Insurance_test$Losses)
mean((testdata - tlogloss)^2)        #MSE from 20% sample
mean((modeltwo_summary$residuals)^2)
```

### We can see that the MSE of in-sample and out-of-sample are very close, which further proves our model 2 is a good fit.  

# Diagnostics Analysis
```{r}
par(mfrow=c (1, 2)) 
plot(model_2)              
```
# Limitation: no linear relationship within age groups. 
```{r}
model_a<- lm(logloss ~Age, subset=Age.Band=="16-25")
summary(model_a)
```
```{r}
model_b<- lm(logloss ~Age, subset=Age.Band=="26-59")
summary(model_b)
```
```{r}
model_c<- lm(logloss ~Age, subset=Age.Band=="60+")
summary(model_c)
```
```