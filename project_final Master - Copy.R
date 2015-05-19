# GROUP 7 - AUTO INSURANCE - ANALYSIS & PREDICTION

Regression and Variable Selection
=================================================
  
  The objective of this project is to analyze historical
auto insurance loss data and predict insurance company's 
future losses based on customer demographics.


Auto Insurance Data
===================
### LOAD LIBRARY
library("lmtest")
library("sandwich")
###READ THE DATASET
rm(list=ls())
InsuranceData=read.csv(file="Insurance Data Updated.csv")
attach(InsuranceData)

The original data are 15,290 observations on 8 variables,
losses being the response variable $y$.

Variable Description
----------------------
Age | Age of policy holder
Years of Driving Experience | Years driven 
Number of Vehicles  | Number of Cars insured under policy 
Gender | M/F
Married | Married/Single
Vehicle Age | Age of vehicle insured under policy
Fuel Type | Petro/Diesel
Losses | Loss claimed under policy


### sampling (Split Dataset Randomly)
Next we sample 90% of the original data and use it 
as the training set. The remaining 10% is used 
as test set. The regression model will be built 
on the training set and future performance 
of your model will be evaluated with the test set.

```{r}
set.seed(1000)
subset <- sample(nrow(InsuranceData),nrow(InsuranceData)*0.90)
Insurance_train = InsuranceData[subset,]
Insurance_test = InsuranceData[-subset,]
```


Data Analysis and Visualization
----------------------------------
# Histogram of Losses. 

### RESPONSE VARIABLE EXPLORATION

####DISTRIBUTION ANALYSIS
#####PERCENTILES,VARIANCE,FREQ DISTRIBUTION

plot(Losses,
    col=rgb(74,126,187,max=255),
    pch=18,
    ylab="Losses",
    xlab="No. of Records",
#    xlim=range(0:1500)
)

grid(nx=NA,ny=NULL,col=rgb(165,165,165,max=255),lty=1)

boxplot(Losses,
        horizontal=TRUE,
        las=1, #make all labels horizontal
        notch=TRUE, #notches for CI for median
        col=rgb(74,126,187,max=255),
        boxwex=0.5,#width of box as proportion of original
        whisklty=1, #whisker line type; 1=solid line
        staplelty=0, #staple(line at end) type; 0=none
        outpch=16, #symbols for outliers; 16=filled circle
        outcol=rgb(74,126,187,max=255), #colors for outliers
        main="Distribution of Losses",
        xlab="Losses(00s)"
        )
grid(ny=NA,nx=10,col=rgb(165,165,165,max=255),lty=1)

hist(Losses,
    breaks=100, 
    xlab="Loss (00s)",
    ylab="Density of Policy", 
    main="Distribution of Losses",
    col=rgb(218,238,243,max=255),
    border=0,
    prob=TRUE)

curve(dnorm(x,mean=mean(Losses),sd=sd(Losses)),
      col="red",
      lwd=3,
      add=TRUE
      )

lines(density(Losses),col=rgb(0,162,0,max=255),lwd=1)
lines(density(Losses,adjust=3),col=rgb(112,48,160,max=255),lwd=1)

min(Losses)
quantile(Losses,c(0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975,0.99,0.995))
max(Losses)
mean(Losses)
sd(Losses)

####OUTLIER TREATMENT
#####IDENTIFY OUTLIERS/THRESHOLD LIMITS - CAP AT THRESHOLDS

We capped the loss to the 99th Percentile i.e. 1200.

plot(Capped.Losses,
    col=rgb(74,126,187,max=255),
    pch=18,
    ylab="Losses",
    xlab="No. of Records",
#    xlim=range(0:1500)
)

grid(nx=NA,ny=NULL,col=rgb(165,165,165,max=255),lty=1)

boxplot(Capped.Losses,
horizontal=TRUE,
las=1, #make all labels horizontal
notch=TRUE, #notches for CI for median
col=rgb(74,126,187,max=255),
boxwex=0.5,#width of box as proportion of original
whisklty=1, #whisker line type; 1=solid line
staplelty=0, #staple(line at end) type; 0=none
outpch=16, #symbols for outliers; 16=filled circle
outcol=rgb(74,126,187,max=255), #colors for outliers
main="Distribution of Capped.Losses",
xlab="Capped.Losses(00s)"
)
grid(ny=NA,nx=10,col=rgb(165,165,165,max=255),lty=1)

hist(Capped.Losses,
breaks=100, 
xlab="Loss (00s)",
ylab="Density of Policy", 
main="Distribution of Capped.Losses",
col=rgb(218,238,243,max=255),
border=0,
prob=TRUE)

curve(dnorm(x,mean=mean(Capped.Losses),sd=sd(Capped.Losses)),
col="red",
lwd=3,
add=TRUE
)

lines(density(Capped.Losses),col=rgb(0,162,0,max=255),lwd=1)
lines(density(Capped.Losses,adjust=3),col=rgb(112,48,160,max=255),lwd=1)

min(Capped.Losses)
quantile(Capped.Losses,c(0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975,0.99,0.995))
max(Capped.Losses)
mean(Capped.Losses)
sd(Capped.Losses)

# Descriptive- Histogram of Capped.Loss- distribution seems more normal. Capping the losses to remove the outliers hasnt changed the median, though the mean has changed by a small fraction & the standard deviation by around 9%.

*******For project possibly add gg-plots to show better visualization between each variable and losses********

# Descriptive- Plots for each pair, the color is to make the dots look lighter , easier to see
### INDEPENDENT VARIABLE ANALYSIS
#### BIVARIATE ANALYSIS OF RESPONSE AGAINST INDEPENDENT VARIABLES

**Age
AgeProf=data.frame(Age,Losses,Capped.Losses,Count)
AgeProfSumm=ddply(AgeProf, "Age", numcolwise(sum))
AgeProfSumm2=data.frame(AgeProfSumm,
            AvgLosses=round(AgeProfSumm$Losses/AgeProfSumm$Count,digits=0),
            AvgCappedLosses=round(AgeProfSumm$Capped.Losses/AgeProfSumm$Count,digits=0),
            Obs=round(AgeProfSumm$Count/sum(AgeProfSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(AgeProfSumm2$Obs,col="lightgray",las=1,border=0,axes=F)
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(AgeProfSumm2$Age,AgeProfSumm2$AvgLosses,ylim=range(0:600),pch=18, ylab = NA,xlab=NA)
mtext("Age",side=1,line=1.2)
mtext("Mean Loss($)",side=2,line=2.5)
lines(AgeProfSumm2$Age,AgeProfSumm2$AvgLosses, type='l', col="red", lwd=3)
lines(AgeProfSumm2$Age,AgeProfSumm2$AvgCappedLosses, type='l', col="darkgreen", lwd=2)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses","Avg Cap. Losses"), 
      fill=c("lightgray","red","darkgreen"), horiz=TRUE)

**Age Band

AgeBandProf=data.frame(Age.Band,Losses,Capped.Losses,Count)
AgeBandProfSumm=ddply(AgeBandProf, "Age.Band", numcolwise(sum))
AgeBandProfSumm2=data.frame(AgeBandProfSumm,
            AvgLosses=round(AgeBandProfSumm$Losses/AgeBandProfSumm$Count,digits=0),
            AvgCappedLosses=round(AgeBandProfSumm$Capped.Losses/AgeBandProfSumm$Count,digits=0),
            Obs=round(AgeBandProfSumm$Count/sum(AgeBandProfSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(AgeBandProfSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(0:50),width=1)
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(AgeBandProfSumm2$Age.Band,AgeBandProfSumm2$AvgLosses,ylim=range(0:600), ylab = NA,xlab=NA,type="n")
mtext("Age Band",side=1,line=1.7)
mtext("Mean Loss($)",side=2,line=2.5)
lines(AgeBandProfSumm2$Age,AgeBandProfSumm2$AvgLosses, type='l', col="red", lwd=3)
lines(AgeBandProfSumm2$Age,AgeBandProfSumm2$AvgCappedLosses, type='l', col="darkgreen", lwd=2)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses","Avg Cap. Losses"), 
      fill=c("lightgray","red","darkgreen"), horiz=TRUE)

**Years of Driving Experience
YDEProf=data.frame(Years.of.Driving.Experience,Losses,Capped.Losses,Count)
YDEProfSumm=ddply(YDEProf, "Years.of.Driving.Experience", numcolwise(sum))
YDEProfSumm2=data.frame(YDEProfSumm,
            AvgLosses=round(YDEProfSumm$Losses/YDEProfSumm$Count,digits=0),
            AvgCappedLosses=round(YDEProfSumm$Capped.Losses/YDEProfSumm$Count,digits=0),
            Obs=round(YDEProfSumm$Count/sum(YDEProfSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(YDEProfSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(0:10))
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(YDEProfSumm2$Years.of.Driving.Experience,YDEProfSumm2$AvgLosses,ylim=range(0:600),pch=18, ylab = NA,xlab=NA)
mtext("Age",side=1,line=1.2)
mtext("Mean Loss($)",side=2,line=2.5)
lines(YDEProfSumm2$Years.of.Driving.Experience,YDEProfSumm2$AvgLosses, type='l', col="red", lwd=3)
lines(YDEProfSumm2$Years.of.Driving.Experience,YDEProfSumm2$AvgCappedLosses, type='l', col="darkgreen", lwd=2)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses","Avg Cap. Losses"), 
      fill=c("lightgray","red","darkgreen"), horiz=TRUE)

**Gender
GenderProf=data.frame(Gender,Losses,Capped.Losses,Count)
GenderProfSumm=ddply(GenderProf, "Gender", numcolwise(sum))
GenderProfSumm2=data.frame(GenderProfSumm,
            AvgLosses=round(GenderProfSumm$Losses/GenderProfSumm$Count,digits=0),
            AvgCappedLosses=round(GenderProfSumm$Capped.Losses/GenderProfSumm$Count,digits=0),
            Obs=round(GenderProfSumm$Count/sum(GenderProfSumm$Count)*100,digits=2)
            )
par(mar = c(5,5,2,5))
barplot(GenderProfSumm2$Obs,col="lightgray",las=1,border=0,axes=F,ylim=range(49:51))
axis(side=4)
mtext("%Obs",side=4,line=2)
par(new = T)
plot(GenderProfSumm2$Gender,GenderProfSumm2$AvgLosses,ylim=range(0:600),pch=18, ylab = NA,xlab=NA)
mtext("Age",side=1,line=1.2)
mtext("Mean Loss($)",side=2,line=2.5)
lines(GenderProfSumm2$Gender,GenderProfSumm2$AvgLosses, type='l', col="red", lwd=3)
lines(GenderProfSumm2$Gender,GenderProfSumm2$AvgCappedLosses, type='l', col="darkgreen", lwd=2)
par(xpd=T)
legend("bottom", inset=-.22,
      c("% # of Policies","Avg Losses","Avg Cap. Losses"), 
      fill=c("lightgray","red","darkgreen"), horiz=TRUE)
#### HETEROSKEDASTICITY

plot(Capped.Losses ~Vehicle.Age,col="#00000033")
plot(Capped.Losses ~Age,col="#00000033")
plot(Capped.Losses ~Gender,col="#00000033")
plot(Capped.Losses ~Married,col="#00000033")
plot(Capped.Losses ~Fuel.Type,col="#00000033")
plot(Age, Years.of.Driving.Experience,col="#00000033")
plot(Capped.Losses~Age.Band)

#ggplot- Capped Losses by Age Group (note: the y axis scale seems wrong, I need to fix it)
qplot(Capped.Losses, data=InsuranceData, geom="density", fill=Age.Band, alpha=I(.5), 
      main="Distribution of Capped Losses", xlab="Capped.Losses", 
ylab="# of Policy")


#ggplot - Capped Losses by Gender
qplot(Capped.Losses, data=InsuranceData, geom="density", fill=Gender, alpha=I(.5), 
main="Distribution of Capped Losses", xlab="Capped.Losses", 
ylab="# of Policy")

#ggplot-  Histogram- Gender + Age Group vs. Mean Capped Losses
#source: http://stackoverflow.com/questions/17368223/ggplot2-multi-group-histogram-with-in-group-proportions-rather-than-frequency

bar<-ggplot(InsuranceData, aes(Gender, Capped.Losses,fill=Age.Band))
bar+ stat_summary(fun.y=mean,geom="histogram", position="dodge", width=0.2)


bar<-ggplot(insurance.data, aes(Married, Capped.Losses,fill=Age.Band))
bar+ stat_summary(fun.y=mean,geom="histogram", position="dodge", width=0.2)

#Find correlations between variables 
#(I did some research and found that we can only calculate correlations for continuous variables, not discrete ones )
cor(Age,Avg.Age)
cor(Years.of.Driving.Experience,Age)
cor(Vehicle.Age,Avg.Vehicle.Age)
#interesting finding. include the plot in here.  What other correlation would be useful for presentation?

#Age and experience is highliy correlated. therefore we are dropping one of them as our variables.
cor(Age,Years.of.Driving.Experience)


### DATA DICTIONARY
#VARIABLE NAME    VARIABLE DESCRIPTION    VALUES STORED    VARIABLE TYPE

### DATA PROFILING
#TBD


Model Building
------------------

Variable Selection


### MULTIVARIATE LIN REGRESSION
#### FITTING THE REGRESSION
##### MULTICOLLINEARITY CHECK
age vs avg age
veh age vs avg veh age
age vs years of driving exp

##### FIX HETEROSKEDASTICITY

```{r}
FitLinReg=lm(Capped.Losses~Number.of.Vehicles+Avg.Age+Gender.Dummy+Married.Dummy+Avg.Vehicle.Age+Fuel.Type.Dummy,LinRegData)
summary(FitLinReg)
```

The following model includes all $x$ varables in the model
```{r, eval=FALSE}
model_1 <- lm(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat, data=Boston_train)
```

To include all variables in the model, you can write the statement this simpler way.


```{r}
model_1 <- lm(medv~., data=Boston_train)
summary(model_1)
```

Evaluating Model Fitness 
------------
### In-sample model evaluation (train error)
MSE of the regression, which is the square of 'Residual standard error' in the above summary. It is the sum of squared residuals(SSE) divided by degrees of freedom (n-p-1). In some textbooks the sum of squred residuals(SSE) is called residual sum of squares(RSS). MSE of the regression should be the unbiased estimator for variance of $\epsilon$, the error term in the regression model.

```{r}
model_summary <- summary(model_1)
(model_summary$sigma)^2
```

$R^2$ of the model
```{r}
model_summary$r.squared
```


Adjusted-$R^2$ of the model, if you add a variable (or several in a group), SSE will decrease, $R^2$ will increase, but Adjusted-$R^2$ could go either way.
```{r}
model_summary$adj.r.squared
```

AIC and BIC of the model, these are information criteria. Smaller values indicate better fit.

```{r}
AIC(model_1)
BIC(model_1)
```

BIC, AIC, and Adjusted $R^2$ have complexity penalty in the definition, thus when comparing across different models they are better indicators on how well the model will perform on future data.


### Out-of-sample prediction (test error)
To evaluate how the model performs on future data, we use predict() to get the predicted values from the test set.
```{r, eval=FALSE}
#pi is a vector that contains predicted values for test set.
pi <- predict(object = model_1, newdata = Boston_test)
```
Just as any other function, you can write the above statement the following way as long as the arguments are in the right order.


```{r, echo=FALSE}
subset <- sample(nrow(Boston),nrow(Boston)*0.90)
Boston_train = Boston[subset,]
Boston_test = Boston[-subset,]
model_1 <- lm(medv~., data=Boston_train)
```

```{r, eval=TRUE}
pi <- predict(model_1, Boston_test)
```


The most common measure is the Mean Squared Error (MSE): average of the squared differences between the predicted and actual values
```{r}
mean((pi - Boston_test$medv)^2)
```
A less popular measure is the Mean Absolute Error (MAE). You can probably guess that here instead of taking the average of squared error, MAE is the average of absolute value of error.
```{r}
mean(abs(pi - Boston_test$medv))
```


Note that if you ignore the second argument of predict(), it gives you the in-sample prediction on the training set:
```{r, eval=FALSE}
predict(model_1)
```
Which is the same as
```{r, eval=FALSE}
model_1$fitted.values
```


Variable Selection
------------------------
### Compare Model Fit Manually
```{r eval=FALSE}
model_1 <- lm(medv~., data = Boston_train)
model_2 <- lm(medv~crim+zn, data = Boston_train)
summary(model_1)
summary(model_2)
AIC(model_1)
AIC(model_2)
```


### Forward/Backward/Stepwise Regression Using AIC
To perform the Forward/Backward/Stepwise Regression in R, we need to define the starting points:
```{r}
nullmodel=lm(medv~1, data=Boston_train)
fullmodel=lm(medv~., data=Boston_train)
```
nullmodel is the model with no varaible in it, while fullmodel is the model with every variable in it.

#### Backward Elimination
```{r}
model.step<-step(fullmodel,direction='backward')
```

#### Stepwise Selection (Output Omitted)
```{r, eval=FALSE}
model.step<-step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='both')
```

One caution when comparing fit statistics using AIC, the definition varies by program/procedure.
```{r}
extractAIC(model_1)
AIC(model_1)
```


* For pros and cons of variable/model selection using the common fit statistics: (adjusted) $R^2$, MSE, AIC, BIC, etc. refer to Ch9 in "Applied Linear Regression Models" by Kutner et al.
* For other variable selection methods refer to section 3.4 - 3.8 of ["Elements of Statistical Learning" (Free Online)](http://www-stat.stanford.edu/~tibs/ElemStatLearn/).

Diagnostic Plots
-----------------
The diagnostic plots are not as important when regression is used in predictive (supervised) data mining as when it is used in economics. However it is still good to know:

1. What the diagnostic plots should look like when no assumption is violated?

2. If there is something wrong, what assumptions are possibly violated?

3. What implications does it have on the analysis?

4. (How) can I fix it?

Roughly speaking, the table summarizes what you should look for in the following plots

Plot Name  | Good  
------------- | -------------
Residual vs. Fitted  | No pattern, scattered around 0 line
Normal Q-Q | Dots fall on dashed line 
Residual vs. Leverage | No observation with large Cook's distance

```{r}
plot(model_1)
```



