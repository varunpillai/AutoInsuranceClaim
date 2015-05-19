# GROUP 7 - AUTO INSURANCE - ANALYSIS & PREDICTION

=================================================
Objective: 
     The objective of this project is to analyze historical insurance data to find predictive variables that will ultimately predict an insurance company's future losses 

Data dictionary and source

Variable Description
----------------------
Age | Age of policy holder
Age.Band | Age groups of 16-25, 26-59, and 60+
Years of Driving Experience | Years driven 
Number of Vehicles  | Number of Cars insured under policy 
Gender | M/F
Married | Married/Single
Vehicle Age | Age of vehicle insured under policy
Vehicle Age Band | Groups of Vehicle Age
Fuel Type | Petro/Diesel
Losses | Loss claimed under policy

The data come from an auto insurance company in India, where variables and the background are provided.

Dataset summary

===================
### LOAD LIBRARY
```{r}
install.packages("lmtest")
install.packages("sandwich")
install.packages("plyr")
install.packages("ggplot2")
library("lmtest")
library("sandwich")
library("plyr")
library("ggplot2")
```

###READ THE DATASET
```{r}
rm(list=ls())
InsuranceData=read.csv(file="Insurance Data Updated.csv")
```

```{r}
summary(InsuranceData)
```

### Sampling (Split Dataset Randomly)
Next we sample 80% of the original data and use it as the training set. The remaining 20% is used 
as test set. The regression model will be built on the training set and future performance of your model will be evaluated with the test set.

```{r}
set.seed(1000)
subset <- sample(nrow(InsuranceData),nrow(InsuranceData)*0.80)
Insurance_train = InsuranceData[subset,]
Insurance_test = InsuranceData[-subset,]

logloss<- log(Insurance_train$Losses)
InsuranceData=data.frame(Insurance_train,logloss)
attach(InsuranceData)
write.csv(InsuranceData, file = "MyData.csv")
```

#Response Variable Exploration

Plot of Losses
```{r}
plot(Losses,col=rgb(74,126,187,max=255),pch=18,ylab="Losses",xlab="No. of Records")
grid(nx=NA,ny=NULL,col=rgb(165,165,165,max=255),lty=1)
```
Histogram of Losses
```{r}
hist(Losses,breaks=100, xlab="Losses (00s)",ylab="Density of Policy", main="Distribution of Losses",col=rgb(218,238,243,max=255),border=0,prob=TRUE)
curve(dnorm(x,mean=mean(Losses),sd=sd(Losses)),col="red",lwd=2,add=TRUE)
lines(density(Losses,adjust=3),col=rgb(112,48,160,max=255),lwd=2)
```

```{r}
summary(Losses)
quantile(Losses,c(0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975,0.99,0.995))
```

From the plot graph, we can see quite a few outliners. From the histogram, the losses is highly skewed to the right. We determined that this type of distribution will influence our regression model. Therefore, we deciced to normalize the data. 


# Histogram of Losses
```{r}
hist(logloss,breaks=100, xlab="LogLoss",ylab="Density of Policy", main="Distribution of LogLoss",col=rgb(218,238,243,max=255),border=0,prob=TRUE)
curve(dnorm(x,mean=mean(logloss),sd=sd(logloss)),col="red",lwd=2,add=TRUE)
lines(density(logloss,adjust=3),col=rgb(112,48,160,max=255),lwd=2)
```

#Independent Variables Analysis
# Bivariate analysis of response agains independent variables

plot(logloss ~Age,col=rgb(74,126,187,max=255))

**Age
AgeProf=data.frame(Age,logloss,Count)
AgeProfSumm=ddply(AgeProf, "Age", numcolwise(sum))
AgeProfSumm2=data.frame(AgeProfSumm,
            AvgLosses=round(AgeProfSumm$logloss/AgeProfSumm$Count,digits=3),
            Obs=round(AgeProfSumm$Count/sum(AgeProfSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(AgeProfSumm2$Obs,col="lightgray",las=1,border=0,axes=F)
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(AgeProfSumm2$Age,AgeProfSumm2$AvgLosses,ylim=range(4:7),pch=18, ylab = NA,xlab=NA)
mtext("Age",side=1,line=1.2)
mtext("Log Loss($)",side=2,line=2.5)
lines(AgeProfSumm2$Age,AgeProfSumm2$AvgLosses, type='l', col="red", lwd=3)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses"), 
      fill=c("lightgray","red"), horiz=TRUE)
min(AgeProfSumm2$AvgLosses)
plot(logloss ~Age.Band,col=rgb(74,126,187,max=255))

**Age.Band
AgeBand=data.frame(Age.Band,logloss,Count)
AgeBandSumm=ddply(AgeBand, "Age.Band", numcolwise(sum))
AgeBandSumm2=data.frame(AgeBandSumm,
            AvgLosses=round(AgeBandSumm$logloss/AgeBandSumm$Count,digits=3),
            Obs=round(AgeBandSumm$Count/sum(AgeBandSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(AgeBandSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(0:50))
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(AgeBandSumm2$Age,AgeBandSumm2$AvgLosses,ylim=range(4:7),pch=18, ylab = NA,xlab=NA)
mtext("Age",side=1,line=1.7)
mtext("Log Loss($)",side=2,line=2.5)
lines(AgeBandSumm2$Age,AgeBandSumm2$AvgLosses, type='l', col="red", lwd=3)
lines(AgeBandSumm2$Age,AgeBandSumm2$AvgCappedLosses, type='l', col="darkgreen", lwd=2)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses"), 
      fill=c("lightgray","red"), horiz=TRUE)

plot(logloss ~Vehicle.Age.Band,col="#FF2030")

**Vehicle.Age.Band
VehAgeBand=data.frame(Vehicle.Age.Band,logloss,Count)
VehAgeBandSumm=ddply(VehAgeBand, "Vehicle.Age.Band", numcolwise(sum))
VehAgeBandSumm2=data.frame(VehAgeBandSumm,
            AvgLosses=round(VehAgeBandSumm$logloss/VehAgeBandSumm$Count,digits=3),
            Obs=round(VehAgeBandSumm$Count/sum(VehAgeBandSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(VehAgeBandSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(0:50))
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(VehAgeBandSumm2$Vehicle.Age.Band,VehAgeBandSumm2$AvgLosses,ylim=range(0:7),pch=18, ylab = NA,xlab=NA)
mtext("Vehicle Age",side=1,line=1.7)
mtext("Log Loss($)",side=2,line=2.5)
lines(VehAgeBandSumm2$Vehicle.Age.Band,VehAgeBandSumm2$AvgLosses, type='l', col="red", lwd=3)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses"), 
      fill=c("lightgray","red"), horiz=TRUE)

plot(logloss ~Gender,col="#FF2030")

**Gender
GenderList=data.frame(Gender,logloss,Count)
GenderListSumm=ddply(GenderList, "Gender", numcolwise(sum))
GenderListSumm2=data.frame(GenderListSumm,
            AvgLosses=round(GenderListSumm$logloss/GenderListSumm$Count,digits=3),
            Obs=round(GenderListSumm$Count/sum(GenderListSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(GenderListSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(0:60))
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(GenderListSumm2$Gender,GenderListSumm2$AvgLosses,ylim=range(0:7),pch=18, ylab = NA,xlab=NA)
mtext("Gender",side=1,line=1.7)
mtext("Log Loss($)",side=2,line=2.5)
lines(GenderListSumm2$Gender,GenderListSumm2$AvgLosses, type='l', col="red", lwd=3)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses"), 
      fill=c("lightgray","red"), horiz=TRUE)

plot(logloss ~Married,col="#FF2030")

**Marital Status
MaritalList=data.frame(Married,logloss,Count)
MaritalListSumm=ddply(MaritalList, "Married", numcolwise(sum))
MaritalListSumm2=data.frame(MaritalListSumm,
            AvgLosses=round(MaritalListSumm$logloss/MaritalListSumm$Count,digits=3),
            Obs=round(MaritalListSumm$Count/sum(MaritalListSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(MaritalListSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(0:60))
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(MaritalListSumm2$Married,MaritalListSumm2$AvgLosses,ylim=range(0:7),pch=18, ylab = NA,xlab=NA)
mtext("Marital Status",side=1,line=1.7)
mtext("Log Loss($)",side=2,line=2.5)
lines(MaritalListSumm2$Married,MaritalListSumm2$AvgLosses, type='l', col="red", lwd=3)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses"), 
      fill=c("lightgray","red"), horiz=TRUE)

plot(logloss ~Fuel.Type,col="#FF2030")

**Fuel Type
FuelTypeList=data.frame(Fuel.Type,logloss,Count)
FuelTypeListSumm=ddply(FuelTypeList, "Fuel.Type", numcolwise(sum))
FuelTypeListSumm2=data.frame(FuelTypeListSumm,
            AvgLosses=round(FuelTypeListSumm$logloss/FuelTypeListSumm$Count,digits=3),
            Obs=round(FuelTypeListSumm$Count/sum(FuelTypeListSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(FuelTypeListSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(0:80))
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(FuelTypeListSumm2$Fuel.Type,FuelTypeListSumm2$AvgLosses,ylim=range(0:7),pch=18, ylab = NA,xlab=NA)
mtext("FuelType Status",side=1,line=1.7)
mtext("Log Loss($)",side=2,line=2.5)
lines(FuelTypeListSumm2$Fuel.Type,FuelTypeListSumm2$AvgLosses, type='l', col="red", lwd=3)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses"), 
      fill=c("lightgray","red"), horiz=TRUE)

#Age group and Gender
bar<-ggplot(Insurance_train, aes(Gender,logloss,fill=Age.Band))
bar+ stat_summary(fun.y=mean,geom="histogram", position="dodge", width=0.2)

#Age group and Marrital status
bar<-ggplot(Insurance_train, aes(Married, logloss,fill=Age.Band))
bar+ stat_summary(fun.y=mean,geom="histogram", position="dodge", width=0.2)


#Find correlations between variables . Age and Driving experience have 99% correlation.
cor(Years.of.Driving.Experience,Age)
cor(logloss, Age)
cor(logloss,Years.of.Driving.Experience)
cor(logloss,Number.of.Vehicles)

##Anova Test(for unequal variances ) and T-test
#Comparing logloss between different age groups
ageloss<- aov(logloss ~Age.Band)
summary(ageloss)
TukeyHSD(ageloss)

#Comparing logloss between gender
t.test(logloss ~Gender)

#Comparing logloss between fuel type
t.test(logloss ~Fuel.Type)


Model Building and Variable Selection
---------------------------------------
#Initial Model
The following model includes all $x$ varables in the model

model_1 <- lm(logloss ~Number.of.Vehicles+Age:Years.of.Driving.Experience+Gender +Married +Vehicle.Age +Fuel.Type)
summary(model_1)
step(model_1,direction="backward")


Based on the results, the second model has higher AIC which implies better fitness. We then continue to evaluate it based on the value of adjusted R square. 
model_2 <- lm(logloss ~Age:Years.of.Driving.Experience+Gender +Married +Vehicle.Age +Fuel.Type)
summary(model_1)$adj.r.squared
summary(model_2)$adj.r.squared


#In-Sample Prediction
#Out-of-Sample Prediction
#Diagnostics Analysis











