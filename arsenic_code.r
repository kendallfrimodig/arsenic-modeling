KRIGING IN CHAPTER 2
```{R}
###Created by: Claudio Owusu
##Install required packages##
#install.packages("latticeExtra")
#install.packages("lattice")
#install.packages("sp")
#install.packages("splancs")
#install.packages("rgdal")
#install.packages("gstat")
#install.packages("RColorBrewer")
#install.packages("rgeos")
#install.packages("spatstat")
#install.packages("maptools")
#install.packages("GISTools", dependencies = TRUE)
#install.packages("raster")
#install.packages("tmap")
#install.packages ("constrainedKriging") ## back transforms lognormal krigig
#install.packages ("automap")
##load the required spatial libraies
library(RColorBrewer)
library(latticeExtra)
library(splancs)
library(gstat)
library(rgdal)
library(rgeos)
library(spatstat)
library(maptools)
library(GISTools)
library(raster)
library(tmap)
library(constrainedKriging)
library(fishmethods) ## functions to back transform
##set working directory##
setwd("C:/Users/clowu/Documents/UNCC Dissertation/Chapter
1_Geocoding_Manuscript/Data_For_Analysis/Exposure Misclassification")
##load the required datasets for the analysis




reference <-read.csv ("reference.csv", header=TRUE, sep=",")
parceldata <-read.csv ("parceldataF.csv", header=TRUE, sep=",")
addresspointsdata <-read.csv ("addresspointsdataF.csv", header=TRUE, sep=",")
streetdata <-read.csv ("streetdataF.csv", header=TRUE, sep=",")
## creating a spatial objects from the datasets
coordinates(reference) <- ~xloc +yloc
coordinates(addresspointsdata) <- ~xloc +yloc
coordinates(parceldata) <- ~xloc +yloc
coordinates(streetdata) <- ~xloc +yloc
##load the spatial boundary of Gaston
shp <- readOGR(".", "GC_Boundary")
summary(shp)
#Assign a projection from the boundary shapefile (shp) to all the datasets the datasets
proj4string(addresspointsdata) <- CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
plot(addresspointsdata, col='red', cex=0.7)
proj4string(parceldata) <- CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
plot(parceldata, col='black', cex=0.7)
proj4string(streetdata) <- CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m
+no_defs")
plot(streetdata, col='grey', cex=0.7)
# A fuction to develop a grid from a dataset with xyz locations
#Npts is the approximate number of points to generate
##This function was borrowed from Dr. Aston Shortbridge##
build.convex.grid <- function (x, y, npts) {
library(splancs) # for gridding and inout functions
# First make a convex hull border (splancs poly)
ch <- chull(x, y) # index for pts on convex hull
ch <- c(ch, ch[1])
border <- cbind(x[ch], y[ch]) # This works as a splancs poly





# Now fill it with grid points
xy.grid <- gridpts(border, npts)
return(xy.grid)
}
### Create a surface for prediction and visualization from xyz that approximates Gaston
County ###
cm <- coordinates(reference)
grid <- data.frame(build.convex.grid(cm[,1], cm[,2], 20000))
names(grid) <- c('Xloc', 'Yloc')
gridded(grid) <- ~Xloc+Yloc
plot(grid, add=TRUE, pch=1, cex=0.4) # add this to the points plot
##Assign the same projection in the data to the grid
proj4string(grid) <- CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
#Convert the Gaston boundary into SpatialPolygons object and make it the same
projection as the data
shp <- shp@polygons
shp <- SpatialPolygons(shp, proj4string=CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m
+no_defs")) #make sure the shapefile has the same CRS from the data, and from the
prediction grid.
plot(shp)
#Clip the prediction grid with the shapefile
clip_grid <- grid[!is.na(over(grid, shp)),]
plot(clip_grid, add=TRUE, pch=1, cex=0.4)
##CREATE NORTH ARROW AND SCALE BAR FOR THE CHARTS###
l2 = list("SpatialPolygonsRescale", layout.north.arrow(), offset = c(392000,162000),
scale = 2500)
l3 = list("SpatialPolygonsRescale", layout.scale.bar(), offset = c(390000,160000),
scale = 5000, fill=c("transparent","black"))
l4 = list("sp.text", c(390000,161000), "0")
l5 = list("sp.text", c(394000,161000), "5000 m")
######1.ANALYSING THE DATA Geocoded using addresspoints#######




### Variogram analysis
addresspoints_error.vg <- variogram(Error2~1, addresspointsdata, width = 100, cutoff =
2000)
##The output and plot of the observed variogram
addresspoints_error.vg
plot(addresspoints_error.vg)
##Check Anisotrophy using variogram maps
vgm.map1 = variogram(Error2~1, addresspointsdata, width = 100, cutoff = 2000, map =
TRUE)
plot(vgm.map1, threshold = 10)
### Using the Automap to generate the fitting parameters
##library (automap)
#autoAP <- autofitVariogram(Error2~1, addresspointsdata)
#summary(autoAP)
##Choose a best Fitting theoretical variogram using the kappa criteria
options(warn = -1) ##don't print warnings
addresspoints_error.vg.mod <- fit.variogram(addresspoints_error.vg, model=vgm("Sph"))
addresspoints_error.vg.mod
attr(addresspoints_error.vg.mod,"SSEr")
resultsAS<- summary(addresspoints_error.vg.mod)
##Write output to a text
sink("addresspointsmodel.txt")
print(resultsAS)
sink()
#Plot the variogram
png("cex-axis.png")
main<-par(cex.axis= 10.0, cex.lab = 5.0)
plot(addresspoints_error.vg, addresspoints_error.vg.mod, xlab ="Distance (m)",
main=main)
dev.off()
##The predictions
#addresspoints.ok <- krige(Error2~1, addresspointsdata, clip_grid,
model=addresspoints_error.vg.mod, nmax = 5)
#spplot(addresspoints.ok["var1.pred"])
#summary(addresspoints.ok)



##CHECKING for ANISOTRPY
#addresspoints_error.vg.dir=variogram(Error2~1, addresspointsdata, width = 100,
cutoff = 2000,alpha=c(0,45,90,135))
#addresspoints_error.vg.dir
#plot(addresspoints_error.vg.dir)
#addresspoints_error.vg.mod<-fit.variogram(addresspoints_error.vg,model =
vgm(0.033, "Sph", 650, 0.0192, anis=c(45,0.5)))
#plot(addresspoints_error.vg.dir,addresspoints_error.vg.mod, as.table = TRUE)
#addresspoints_error.vg.mod
#attr(addresspoints_error.vg.mod,"SSEr")
### Predicting the addressPointserrors###
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey,
args = list(key = args),
corner = c(0.05,.75))
addresspoints.ok <- krige(Error2~1, addresspointsdata, clip_grid,
model=addresspoints_error.vg.mod, nmax=5)
spplot(addresspoints.ok["var1.pred"],
sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE),
colorkey = list(space = "right", height = 1.0))+
layer(sp.polygons(shp, lwd = 1.5))
summary(addresspoints.ok)
##Export output to raster using the rgdal
writeGDAL(addresspoints.ok["var1.pred"], "pred.addresspointsErr2.tif")
##Export variance to raster using the rgdal
writeGDAL(addresspoints.ok["var1.var"], "variance.addresspointsVar2.tif")
######2.ANALYSING THE DATA Geocoded using Parcel#######
### Variogram analysis
parcel_error.vg <- variogram(Error2~1, parceldata, width = 100, cutoff = 2000)
##The output and plot of the observed variogram
parcel_error.vg
plot(parcel_error.vg)
##Check Anisotrophy using variogram maps
vgm.map2 = variogram(Error2~1, parceldata, width = 100, cutoff = 2000, map = TRUE)
plot(vgm.map2, threshold = 10)



##Choose a best Fitting theoretical variogram using the kappa criteria
options(warn = -1) ##don't print warnings
parcel_error.vg.mod <- fit.variogram(parcel_error.vg, vgm("Sph"))
parcel_error.vg.mod
attr(parcel_error.vg.mod,"SSEr")
resultsAS<- summary(addresspoints_error.vg.mod)
##Write output to a text
sink("parcelsmodel.txt")
print(resultsAS)
sink()
plot(parcel_error.vg, parcel_error.vg.mod, xlab ="Distance (m)")##, cex.axis= 2.0,
cex.lab = 2.0)
### Predicting the parcelserrors###
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey,
args = list(key = args),
corner = c(0.05,.75))
parcel.ok <- krige(Error2~1, parceldata, clip_grid, model=parcel_error.vg.mod, nmax=5)
spplot(parcel.ok["var1.pred"],
sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE),
colorkey = list(space = "right", height = 1.0))+
layer(sp.polygons(shp, lwd = 1.5))
summary(parcel.ok)
##Export output to raster using the rgdal
writeGDAL(parcel.ok["var1.pred"], "pred.parcelErr2.tif")
##Export variance to raster using the rgdal
writeGDAL(parcel.ok["var1.var"], "variance.parcelVar2.tif")
######3.ANALYSING THE DATA Geocoded using street#######
### Variogram analysis
street_error.vg <- variogram(Error2~1, streetdata, width = 100, cutoff = 2000)
##The output and plot of the observed variogram
street_error.vg
plot(street_error.vg)



##Check Anisotrophy using variogram maps
vgm.map = variogram(Error2~1, streetdata, width = 100, cutoff = 2000, map = TRUE)
plot(vgm.map, threshold = 10)
##Choose a best Fitting theoretical variogram using the kappa criteria
options(warn = -1) ##don't print warnings
street_error.vg.mod <-fit.variogram(street_error.vg,vgm("Sph"))
street_error.vg.mod
attr(street_error.vg.mod,"SSEr")
resultsAS<- summary(street_error.vg.mod)
##Write output to a text
sink("streetmodel.txt")
print(resultsAS)
sink()
jpeg(file="streetMod.jpg",bg="white", res=300, pointsize = 16, width = 1200, height =
1200, quality = 100)
plot(street_error.vg, plot.number=F, model = street_error.vg.mod, ylim=c(0.04, 0.08), col
="black", cex.axis = 1.5)
##Plot semivariogram
plot(street_error.vg, street_error.vg.mod, xlab ="Distance (m)", cex.axis= 0.7, cex.lab =
1.5, font.axis = 3)
### Predicting the streeterrors###
### Predicting the parcelserrors###
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey,
args = list(key = args),
corner = c(0.05,.75))
parcel.ok <- krige(Error2~1, parceldata, clip_grid, model=parcel_error.vg.mod, nmax=5)
spplot(parcel.ok["var1.pred"],
sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE),
colorkey = list(space = "right", height = 1.0))+
layer(sp.polygons(shp, lwd = 1.5))
summary(parcel.ok)
### Predicting the streeterrors###
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey,
args = list(key = args),



124
corner = c(0.05,.75))
street.ok <- krige(Error2~1, streetdata, clip_grid, model=street_error.vg.mod, nmax=5)
spplot(street.ok["var1.pred"],
sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE),
colorkey = list(space = "right", height = 1.0))+
layer(sp.polygons(shp, lwd = 1.5))
##Export output to raster using the rgdal
streetPre<-writeGDAL(street.ok["var1.pred"], "pred.streetErr2.tif")
#StreetPre2 <-raster(streetPre)
#streetpre3 <-bt.log()
##Export variance to raster using the rgdal
writeGDAL(street.ok["var1.var"], "variance.streetVar2.tif")



APPENDIX 2: R CODES FOR AUTOLOGISTIC MODEL FOR CHAPTER 3
##ARSENIC DATA ANALYSIS
```{r}
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
```
```{r}
setwd("C:/Users/clowu/Documents/UNCC Dissertation/Chapter
2_Arsenic/Arsenic_Analysis/StatisticalModels_New")
mydata <- read.csv("Final_Arsenic_DataNew2.csv", sep = ',')
colnames(mydata)
##select only the variables important for the modelling
ArsenicData = subset(mydata, select = c(1,14,27:37))
head(ArsenicData)
###transform the data into factors
ArsenicData$Arsenic_Detect2 = as.factor(ArsenicData$Arsenic_Detect2)
levels(ArsenicData$Arsenic_Detect2) = c('No','Yes')
colnames(ArsenicData)
###transform categorical variables into factors
ArsenicData$Depth = as.factor(ArsenicData$Depth)
ArsenicData$Bedrock = as.factor(ArsenicData$Bedrock)
ArsenicData$BedrockNew = as.factor(ArsenicData$BedrockNew)
attach(ArsenicData)
head(ArsenicData)
colnames(ArsenicData)

##Summary table
```{r}
summaryvar = summary(ArsenicData)
summaryvar
write.csv(summaryvar, file = "summary.csv")
```
**Plotting histograms of the variables**
```{r}
pHplot = ggplot(ArsenicData, aes(x=pH))+ geom_histogram(binwidth=1, bins = 14,
color="darkblue", fill="skyblue3")+ scale_x_continuous(name="pH Level", breaks =
c(0,2, 4, 6, 8, 10, 12,14)) + scale_y_continuous(name="Frequency")+ theme_pubclean()
pHplot
Rocktypeplot = ggplot(ArsenicData, aes(Bedrock)) + geom_bar(color="darkblue",
fill="skyblue3") + theme_pubclean() + labs(y="Frequency")
Rocktypeplot
Depthplot = ggplot(ArsenicData, aes(Depth)) + geom_bar(color="darkblue",
fill="skyblue3") + theme_pubclean() + labs(y="Frequency")
Depthplot
colnames(ArsenicData)
```
**Model Development**
```{r}
##Ordinary logistic regression
lmodel =logistic.regression(ArsenicData, y = 'Arsenic_Detect', x =
c('BedrockNew','Depth', 'pH'), penalty = TRUE)
lmodel$model
lmodel$diagTable
lmodel$coefTable
lmodel_pred = predict(lmodel$model, type = 'fitted.ind')



##Spatial autologistic regression
coordinates(ArsenicData) = ~xloc + yloc
lmodel2 = logistic.regression(ArsenicData, y = 'Arsenic_Detect', x =
c('BedrockNew','pH','Depth'), autologistic = TRUE, coords =
coordinates(ArsenicData),longlat = FALSE, penalty = TRUE)
lmodel2$model
lmodel2$diagTable
lmodel2$coefTable
lmodel2$bandwidth
lmodel2_pred = predict(lmodel2$model, type = 'fitted.ind')
autocovariate = lmodel2$AutoCov
residuals2 = lmodel2$Residuals
##write results to csv for mapping
model_arsenic = data.frame(DataID,lmodel2_pred, Arsenic_Detect,autocovariate,
residuals2, xloc, yloc)
write.csv(model_arsenic, file = "model_arsenic.csv")
```
##ROC Curve
```{r}
#install.packages("ROCR")
library(ROCR)
library(pROC)
arsenic_chk = ArsenicData$Arsenic_Detect2
#par(pty ="s")
#roc(arsenic_chk, lmodel_pred, plot = TRUE, legacy.axes = TRUE, xlab = "1-specificity
(False positive rate)", ylab = "Sensitivity (True positive rate)", col="#de2d26", lwd=1,
print.auc = TRUE)
#plot.roc(arsenic_chk, lmodel2_pred,col="#377eb8", lwd=1, print.auc = TRUE, add=
TRUE, print.auc.y=0.4)
#legend("bottomright", legend = c("non-spatial", "spatial"), col =
c("#de2d26","#377eb8" ), lwd = 1)
##Area Under the Receiver Operator Characteristic Curve (AUROC)
chkroc1 = pROC::roc(ArsenicData$Arsenic_Detect2,lmodel2_pred)
chkroc1


ci.auc(chkroc1)
##Checking accuracy of predictions
lmodel2_pred0 = rep("No",990)
lmodel2_pred0[lmodel2_pred>.5] ="Yes"
table(lmodel2_pred0, ArsenicData$Arsenic_Detect2)
mean(lmodel2_pred0==ArsenicData$Arsenic_Detect2)
summary(ArsenicData$Arsenic_Detect2)
##Checking the mean squared errors for the testing data set
n0=length(ArsenicData$Arsenic_Detect)
sse10 = sum((ArsenicData$Arsenic_Detect - lmodel2_pred)^2)
mse10 = sse10 / (n0 - 2)
mse10




APPENDIX 3: R CODES FOR KRIGING MAPS IN CHAPTER 3
###KRIGING ANALYSIS###
```{r}
##Install required packages##
#install.packages("latticeExtra")
#install.packages("lattice")
#install.packages("splancs")
#install.packages("rgdal")
#install.packages("gstat")
#install.packages("RColorBrewer")
#install.packages("rgeos")
#install.packages("spatstat")
#install.packages("maptools")
#install.packages("GISTools", dependencies = TRUE)
#install.packages("raster")
#install.packages("tmap")
#install.packages("sf")
##load the required spatial libraies
library(RColorBrewer)
library(latticeExtra)
library(splancs)
library(gstat)
library(rgdal)
library(rgeos)
library(spatstat)
library(maptools)
library(GISTools)
library(raster)
library(tmap)
library(sf)




**3.Spatial autologistic regression probs interpolation **
```{r}
modelresults = read.csv("model_arsenic.csv", sep = ",")
## creating a spatial objects from the datasets
coordinates(modelresults) <- ~xloc +yloc



setwd("C:/Users/clowu/Documents/UNCC Dissertation/Chapter
2_Arsenic/Arsenic_Analysis/StatisticalModels_New/Kriging")
##load the required datasets for the analysis
addresspoints <-read.csv ("addresspoints.csv", header=TRUE, sep=",")
## creating a spatial objects from the datasets
coordinates(addresspoints) <- ~xloc +yloc
#Assign a projection from the boundary shapefile (shp) to all the datasets the datasets
proj4string(modelresults) <- CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
##load the spatial boundary of Gaston
shp <- readOGR(".", "GC_BoundaryF")
summary(shp)
# A fuction to develop a grid from a dataset with xyz locations
#Npts is the approximate number of points to generate
build.convex.grid <- function (x, y, npts) {
library(splancs) # for gridding and inout functions
# First make a convex hull border (splancs poly)
ch <- chull(x, y) # index for pts on convex hull
ch <- c(ch, ch[1])
border <- cbind(x[ch], y[ch]) # This works as a splancs poly
# Now fill it with grid points
xy.grid <- gridpts(border, npts)
return(xy.grid)
}
### Create a surface for prediction and visualization from xyz that approximates Gaston
County ###
cm <- coordinates(addresspoints)
grid <- data.frame(build.convex.grid(cm[,1], cm[,2], 10000))
names(grid) <- c('Xloc', 'Yloc')
gridded(grid) <- ~Xloc+Yloc
plot(modelresults, col='blue', cex=0.7)
plot(grid, add=TRUE, pch=1, cex=0.4) # add this to the points plot



##Assign the same projection in the data to the grid


proj4string(grid) <- CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
#Convert the Gaston boundary into SpatialPolygons object and make it the same
projection as the data
shp <- shp@polygons
shp <- SpatialPolygons(shp, proj4string=CRS("+proj=lcc +lat_1=34.33333333333334
+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384
+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")) #make sure the
shapefile has the same CRS from the data, and from the prediction grid.
plot(shp)
#Clip the prediction grid with the shapefile
clip_grid <- grid[!is.na(over(grid, shp)),]
plot(clip_grid, add=TRUE, pch=1, cex=0.4)
##CREATE NORTH ARROW AND SCALE BAR FOR THE CHARTS###
l2 = list("SpatialPolygonsRescale", layout.north.arrow(), offset = c(392000,162000),
scale = 2500)
l3 = list("SpatialPolygonsRescale", layout.scale.bar(), offset = c(390000,160000),
scale = 5000, fill=c("transparent","black"))
l4 = list("sp.text", c(390000,161000), "0")
l5 = list("sp.text", c(394000,161000), "5000 m")
```
#Semivariogram for predicted probabilities in the model
```{r}
modelresults$probs <- (modelresults$lmodel2_pred) * 1
PredMod<- variogram(probs~1, modelresults,boundaries = seq(0,20000, l=51))
##The output and plot of the observed variogram
PredMod
plot(PredMod)
PredMod.vg.mod= fit.variogram(PredMod, vgm(c("Gau", "Exp","Sph")))
PredMod.vg.mod
plot(PredMod, model=PredMod.vg.mod)



```{r}
### Local neighborhood Indicator OK on high counts of positive test of coliform ###
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey,
args = list(key = args),
corner = c(0.05,.75))
SpatialAuto.pred <- krige(probs~1, modelresults, clip_grid, model=PredMod.vg.mod,
nmax = 3)
spplot(SpatialAuto.pred["var1.pred"],
sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE),
colorkey = list(space = "right", height = 1.0), at = seq(0, 1, .01))+
layer(sp.polygons(shp, lwd = 1.5))
summary(SpatialAuto.pred)
##Export output to raster using the rgdal
writeGDAL(SpatialAuto.pred["var1.pred"], "pred.SpatialLogNew.tif")
##Export variance to raster using the rgdal
writeGDAL(SpatialAuto.pred["var1.var"], "variance.SpatialLogNew.tif")
```
#Semivariogram for autocovariate variable in the model
```{r}
modelresults$autocov <- (modelresults$autocovariate) * 1
autocovMod<- variogram(autocov~1, modelresults,boundaries = seq(0,20000, l=51))
##The output and plot of the observed variogram
autocovMod
plot(autocovMod)
autocovMod.vg.mod= fit.variogram(autocovMod, vgm(c("Gau", "Exp","Sph")))
autocovMod.vg.mod
plot(autocovMod, model=autocovMod.vg.mod)
```
```{r}
### Predicting the probabilities ###
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey,
args = list(key = args)




corner = c(0.05,.75))
autocov.pred <- krige(autocov~1, modelresults, clip_grid, model=autocovMod.vg.mod,
nmax = 3)
spplot(autocov.pred["var1.pred"],
sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE),
colorkey = list(space = "right", height = 1.0), at = seq(0, 1, .01))+
layer(sp.polygons(shp, lwd = 1.5))
summary(autocov.pred)
##Export output to raster using the rgdal
writeGDAL(autocov.pred["var1.pred"], "pred.autocovNew.tif")
##Export variance to raster using the rgdal
writeGDAL(autocov.pred["var1.var"], "variance.autocovNew.tif")
```
#Semivariogram for residuals in the model
```{r}
modelresults$residualterm <- (modelresults$res)
residualtermMod<- variogram(residualterm~1, modelresults,boundaries = seq(0,20000,
l=51))
##The output and plot of the observed variogram
residualtermMod
plot(residualtermMod)
residualtermMod.vg.mod= fit.variogram(residualtermMod, vgm(c("Gau", "Exp","Sph")))
residualtermMod.vg.mod
plot(residualtermMod, model=residualtermMod.vg.mod)
```
##Ordinary kriging of the model residuals
```{r}
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey,
args = list(key = args),
corner = c(0.05,.75))
residualterm.pred <- krige(residualterm~1, modelresults, clip_grid,
model=residualtermMod.vg.mod, nmax = 5),


spplot(residualterm.pred["var1.pred"],
sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE),
colorkey = list(space = "right", height = 1.0), at = seq(0, 1, .01))+
layer(sp.polygons(shp, lwd = 1.5))
summary(residualterm.pred)
##Export output to raster using the rgdal
writeGDAL(residualterm.pred["var1.pred"], "pred.residualterm.tif")
##Export variance to raster using the rgdal
writeGDAL(residualterm.pred["var1.var"], "variance.residualterm.tif")
```
APPENDIX 4: R CODES FOR MULTIVARIATE LOGISTIC REGRESSION
MODELS IN CHAPTER 4
#LOGISTIC REGRESSION
##COLIFORM BACTERIA DATA ANALYSIS
```{r}
#install.packages("sjPlot")
#install.packages("tidyr")
#install.packages("caret")
#install.packages("jtools")
#install.packages("ggsci")
#install.packages("pROC")
#install.packages("ggpubr")
library(ggpubr)
library(pROC)
library(ggsci)
library(jtools)
library(caret)
library(tidyr)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
#install.packages("car")
#install.packages("aod")
#install.packages("ggplot2")
#install.packages("sp")
#install.packages("lctools")
#install.packages("spdep")
#install.packages("spatialEco")
#install.packages("caret")



library(caret)
library(aod)
library(ggplot2)
library(sp)
#library(lctools)
library(spdep)
library(spatialEco)
library(car)
```
```{r}
##Set the working directory & read the file
setwd("C:/Users/clowu/Documents/UNCC Dissertation/Chapter 3_Total
Coliform/Coliform_Analysis/StatisticalModelNew")
#setwd("E:/StatisticalModelNew")
mydataAll <- read.csv("PathogenData.csv", sep = ',')
colnames(mydataAll)
##select only the variables important for the modelling
PathogenData = subset(mydataAll, select = c(1,19:30))
attach(PathogenData)
head(PathogenData)
###transform categorical data into factors
##1. pathogen
PathogenData$Pathogen2 <-as.factor(PathogenData$Pathogen2)
levels(PathogenData$Pathogen2 )<-c('No','Yes')
head(PathogenData)
##2.welltype
PathogenData$WellType = as.factor(PathogenData$WellType)
head(PathogenData)
##3. Septic Tank Absorption Field
PathogenData$SepTankAF = as.factor(PathogenData$SepTankAF)
head(PathogenData)
##4. MUSYM
PathogenData$MUSYM = as.factor(PathogenData$MUSYM)
head(PathogenData)
PathogenData$CatWellDepth = cut(WellDepth, breaks = c(0, 150, 300, 1025 ), labels =
c("1", "2", "3"))
PathogenData$CatWellDepth = as.factor(PathogenData$CatWellDepth)


#cut(WellDepth, breaks = c(0, 150, 300, 1025 ), labels = c("1", "2", "3"))
PathogenData$RatioDepthCasing1 = (CasingDepth/WellDepth) ## ratio of the casing to
the well depth
attach(PathogenData)
```
**Summary and correlation tables**
```{r}
summaryvar = summary(PathogenData)
summaryvar
#write.csv(summaryvar, file = "summary.csv")
colnames(PathogenData)
correlationtable = round (cor(PathogenData[,c(5:9)]), 3)
correlationtable
#write.csv(correlationtable, file = "correlationtable.csv")
```
**Plotting histograms of the variables**
```{r}
par(mfrow = c(3,2))
hist(Age, xlab = "Age (years)", main = " Histogram of age of well", col='skyblue3')
hist(WellDepth , xlab = "Well Depth(ft)", main = " Histogram of Well depth",
col='skyblue3')
hist(ParcelSize, xlab = "parcel size (acres)", main = " Histogram of parcel size (acres)",
col='skyblue3')
hist(CasingDepth, xlab = "Casing depth", main = " Histogram of Casing depth",
col='skyblue3')
#barplot(prop.table(table(WellType)))
#ggplot(mydata, aes(x=WellType))
```
**perform significant testing categorical variables**
```{r}
##1.Well type
#create a contigency table


WellType_Chisq = table(PathogenData$WellType, PathogenData$Pathogen2)
WellType_Chisq
#Chi-squared test
chisq.test(WellType_Chisq)
##2.Septic tank absorption field rating from USDA
#create a contigency table
SepTankAF_Chisq = table(PathogenData$SepTankAF, PathogenData$Pathogen2)
SepTankAF_Chisq
#Chi-squared test
chisq.test(SepTankAF_Chisq)
#create a contigency table
#PathogenData$CatWellDepth = as.factor(PathogenData$CatWellDepth)
#CatWellType_Chisq = table(PathogenData$CatWellDepth, PathogenData$Pathogen2)
#CatWellType_Chisq
#Chi-squared test
#chisq.test(CatWellType_Chisq)
```
**perform siginificant independent t-test of means for continuous variables**
```{r}
##perform Welch Two Sample t-test of continuous variables
t.test(WellDepth ~ Pathogen1, data = PathogenData)
t.test(RatioDepthCasing1 ~ Pathogen1, data = PathogenData)
t.test(Age ~ Pathogen1, data = PathogenData)
t.test(ParcelSize ~ Pathogen1, data = PathogenData)
```
```{r}
##randomized the data sample
#set.seed(123468)
set.seed(123689)
##split the data into partition
partitionRule <- createDataPartition(PathogenData$Pathogen2, p = 0.8, list = F)
trainingSet <- PathogenData[partitionRule,]
testingSet <- PathogenData[-partitionRule,]
summary(trainingSet)



**Using the the logit model**
```{r}
##Prediction with logistic regression
model1 = glm(Pathogen1 ~ WellType + RatioDepthCasing1 + Age + ParcelSize +
SepTankAF, data=trainingSet, family = "binomial")
summary(model1)
tab_model(model1)
round(exp(coef(model1)),3) ## odds ratios only
round(exp(confint(model1)),3) ##CI for odds ratio
summary(model1)$coef
model1.probs = predict(model1, type = "response") ##predicted probabilities
AIC(model1)
resi.model1 = residuals(model1)
##Check multicolinearity
viftable =round(vif(model1), 3)
viftable
write.csv(viftable, file = "viftable.csv")
```
**Model Validation**
```{r}
##for testing set
summary(testingSet)
model1.probs = as.numeric(unlist(predict(model1, testingSet,type = "response")))
##accuracy check for testing data set
model1.pred = rep("No",231)
model1.pred[model1.probs>.5] ="Yes"
table(model1.pred, testingSet$Pathogen2)
mean(model1.pred==testingSet$Pathogen2)
##Area Under the Receiver Operator Characteristic Curve (AUROC)
chkroc1_test = pROC::roc(testingSet$Pathogen2,model1.probs)
chkroc1_test
###For training set
summary(trainingSet)
model1.probs2 = as.numeric(unlist(predict(model1, trainingSet,type = "response")))
##accuracy check for testing data set

model1.pred2 = rep("No",932)
model1.pred2[model1.probs2<.5] ="Yes"
table(model1.pred2, trainingSet$Pathogen2)
mean(model1.pred2==trainingSet$Pathogen2)
##Area Under the Receiver Operator Characteristic Curve (AUROC)
chkroc1_train = pROC::roc(trainingSet$Pathogen2,model1.probs2)
chkroc1_train
residual.model1 = as.numeric(unlist(residuals(model1)))
```
**Model diagnostics**
```{r}
##Make predictions for total data sets
model1.probsF = as.numeric(unlist(predict(model1, PathogenData,type = "response")))
resi.Model1 = PathogenData$Pathogen1 - model1.probsF
model1_diagnostics = data.frame(SortID,model1.probsF,Pathogen1, Pathogen2,
resi.Model1,WellType, WellDepth, CasingDepth,RatioDepthCasing1, Age, SepTankAF,
ParcelSize, xCoord, yCoord)
write.csv(model1_diagnostics, file = "model1_diagnostics.csv")
```
##plot the prediction with significant variables
```{r}
ModelPred = read.csv("model1_diagnostics.csv",sep = ",", header = TRUE)
attach(ModelPred)
##probablities and well type
welltype_boxplot = ggplot(ModelPred, aes(x=WellType, y= model1.probsF))+
geom_boxplot(aes(fill=WellType)) + scale_color_gradientn(colors = c("#00AFBB",
"#E7B800", "#FC4E07"))+ theme_classic()+theme(legend.position = "none") +
theme(legend.position = "none")+ labs(y="")
welltype_boxplot
Age_pointplot = ggplot(ModelPred, aes(x=Age, y=model1.probsF, colour =
model1.probsF))+ geom_point(size = 3, alpha = 0.6)+ scale_color_gradientn(colors =
c("#00AFBB", "#E7B800", "#FC4E07"))+ theme_classic()+theme(legend.position =
"none") + theme(legend.position = "top")+ labs(y="")
Age_pointplot


RatioDepthCasing1_pointplot = ggplot(ModelPred, aes(x=RatioDepthCasing1,
y=model1.probsF, colour = model1.probsF))+ geom_point(size = 3, alpha = 0.6)+
scale_color_gradientn(colors = c("#00AFBB", "#E7B800", "#FC4E07"))+
theme_classic()+theme(legend.position = "none") + theme(legend.position = "top")+
labs(y="")
RatioDepthCasing1_pointplot
```
##plot the model residuals
```{r}
residmodel1 = read.csv("model1_diagnostics.csv")
summary(residmodel1)
ggplot(residmodel1, aes(xCoord, yCoord, colour =resi.Model1 )) +
viridis::scale_color_viridis()+
geom_point(size = 3)
```
***2. FOR DRILLED WELLS***
```{r}
##Set the working directory & read the file
setwd("C:/Users/clowu/Documents/UNCC Dissertation/Chapter 3_Total
Coliform/Coliform_Analysis/StatisticalModelNew/DrilledWells")
#setwd("E:/StatisticalModelNew/DrilledWells")
myDrilledWells <- read.csv("DrilledWells.csv", sep = ',')
colnames(myDrilledWells)
##select only the variables important for the modelling
myDrilledWells = subset(myDrilledWells, select = c(1,18:29))
attach(myDrilledWells)
head(myDrilledWells)
###transform categorical data into factors
##1. pathogen
myDrilledWells$Pathogen2 <-as.factor(myDrilledWells$Pathogen2)
levels(myDrilledWells$Pathogen2 )<-c('No','Yes')
head(myDrilledWells)
##2.welltype
#PathogenData$WellType = as.factor(PathogenData$WellType)
#head(PathogenData)


##3. Septic Tank Absorption Field
myDrilledWells$SepTankAF = as.factor(myDrilledWells$SepTankAF)
head(myDrilledWells)
myDrilledWells$RatioDepthCasing2 = CasingDepth/WellDepth## ratio of the casing to
the well depth
attach(myDrilledWells)
```
**Summary and correlation tables**
```{r}
summaryvar = summary(myDrilledWells)
summaryvar
#write.csv(summaryvar, file = "summary.csv")
colnames(myDrilledWells)
correlationtable = round (cor(myDrilledWells[,c(5:9)]), 3)
correlationtable
#write.csv(correlationtable, file = "correlationtable.csv")
```
**perform significant testing categorical variables**
```{r}
##1.Septic tank absorption field rating from USDA
#create a contigency table
SepTankAF_Chisq = table(myDrilledWells$SepTankAF, myDrilledWells$Pathogen2)
SepTankAF_Chisq
#Chi-squared test
chisq.test(SepTankAF_Chisq)
```
**perform siginificant independent t-test of means for continuous variables**
```{r}
##perform Welch Two Sample t-test of continuous variables
t.test(RatioDepthCasing2 ~ Pathogen1, data = myDrilledWells)
t.test(Age ~ Pathogen1, data = myDrilledWells)
t.test(ParcelSize ~ Pathogen1, data = myDrilledWells)

```{r}
##randomized the data sample
#set.seed(123468)
set.seed(123689)
##split the data into partition
partitionRule <- createDataPartition(myDrilledWells$Pathogen2, p = 0.8, list = F)
trainingSet1 <- myDrilledWells[partitionRule,]
testingSet1 <- myDrilledWells[-partitionRule,]
summary(trainingSet1)
```
**Aply Prior and bias correction becasue of small number of events than non-events**
```{r}
model2<- glm(Pathogen1 ~ RatioDepthCasing2 + Age + ParcelSize + SepTankAF,
data=trainingSet1, family = "binomial")
summary(model2)
tab_model(model2)
round(exp(coef(model2)),3) ## odds ratios only
round(exp(confint(model2)),3) ##CI for odds ratio
summary(model1)$coef
model2.probs = predict(model2, type = "response") ##predicted probabilities
AIC(model2)
resi.model2 = residuals(model2)
##Check multicolinearity
viftable2 =round(vif(model2), 3)
viftable2
write.csv(viftable2, file = "viftable2.csv")
```
**Model Validation**
```{r}
summary(testingSet1)
probsModel2 = as.numeric(unlist(predict(model2, testingSet1,type = "response")))
##accuracy check for testing data set
model2.pred = rep("No",218)
model2.pred[probsModel2>.5] ="Yes"
table(model2.pred, testingSet1$Pathogen2)
mean(model2.pred==testingSet1$Pathogen2)
##Area Under the Receiver Operator Characteristic Curve (AUROC)



chkroc2_test = pROC::roc(testingSet1$Pathogen2,probsModel2)
chkroc2_test
resiModel2 = as.numeric(unlist(residuals(model2)))
##correct classified for traininfset1
summary(trainingSet1)
probsModel2_train = as.numeric(unlist(predict(model2, trainingSet1,type = "response")))
##accuracy check for testing data set
model2.pred_train = rep("No",873)
model2.pred_train[probsModel2_train>.5] ="Yes"
table(model2.pred_train, trainingSet1$Pathogen2)
mean(model2.pred_train==trainingSet1$Pathogen2)
##Area Under the Receiver Operator Characteristic Curve (AUROC)
chkroc2_train = pROC::roc(trainingSet1$Pathogen2,probsModel2_train)
chkroc2_train
```
**Model diagnostics**
```{r}
##Make predictions for total data sets
##making prediction with trainig data set
model2ProbsF = as.numeric(unlist(predict(model2, myDrilledWells,type = "response")))
resiModel2 = myDrilledWells$Pathogen1 - model2ProbsF
model2_diagnostics = data.frame(SortID,model2ProbsF,Pathogen1, Pathogen2,
resiModel2, WellDepth, CasingDepth,RatioDepthCasing2, Age, SepTankAF, ParcelSize,
xCoord, yCoord)
write.csv(model2_diagnostics, file = "model2_diagnostics.csv")
```
```{r}
d = read.csv("model2_diagnostics.csv")
summary(d)
ggplot(d, aes(xCoord, yCoord, colour =resiModel2 )) +
viridis::scale_color_viridis()+
geom_point(size = 3)