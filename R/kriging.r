# KRIGING ANALYSIS

# Install required packages
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

# load the required spatial libraies
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



#Spatial autologistic regression probs interpolation
modelresults = read.csv("model_arsenic.csv", sep = ",")

# creating a spatial objects from the datasets
coordinates(modelresults) <- ~xloc +yloc



setwd("C:/Users/")

# load the required datasets for the analysis
addresspoints <-read.csv ("addresspoints.csv", header=TRUE, sep=",")

# creating a spatial objects from the datasets
coordinates(addresspoints) <- ~xloc +yloc

# Assign a projection from the boundary shapefile (shp) to all the datasets
proj4string(modelresults) <- CRS("+proj=lcc +lat_1=34.33333333333334+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")

# load the spatial boundary of Gaston
shp <- readOGR(".", "GC_BoundaryF")
summary(shp)

# A fuction to develop a grid from a dataset with xyz locations
# Npts is the approximate number of points to generate
build.convex.grid <- function (x, y, npts) {
# for gridding and inout functions
library(splancs) 
# First make a convex hull border (splancs poly)
# index for pts on convex hull
ch <- chull(x, y) 
ch <- c(ch, ch[1])
# This works as a splancs poly
border <- cbind(x[ch], y[ch]) 
# Now fill it with grid points
xy.grid <- gridpts(border, npts)
return(xy.grid)
}


# Create a surface for prediction and visualization from xyz that approximates Gaston County 
cm <- coordinates(addresspoints)
grid <- data.frame(build.convex.grid(cm[,1], cm[,2], 10000))
names(grid) <- c('Xloc', 'Yloc')
gridded(grid) <- ~Xloc+Yloc
plot(modelresults, col='blue', cex=0.7)
plot(grid, add=TRUE, pch=1, cex=0.4) # add this to the points plot



# Assign the same projection in the data to the grid
proj4string(grid) <- CRS("+proj=lcc +lat_1=34.33333333333334+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")


# Convert the Gaston boundary into SpatialPolygons object and make it the same projection as the data
shp <- shp@polygons
shp <- SpatialPolygons(shp, proj4string=CRS("+proj=lcc +lat_1=34.33333333333334+lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2192024384+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")) 


# make sure theshapefile has the same CRS from the data, and from the prediction grid.
plot(shp)


# Clip the prediction grid with the shapefile
clip_grid <- grid[!is.na(over(grid, shp)),]
plot(clip_grid, add=TRUE, pch=1, cex=0.4)


# CREATE NORTH ARROW AND SCALE BAR FOR THE CHARTS###
l2 = list("SpatialPolygonsRescale", layout.north.arrow(), offset = c(392000,162000), scale = 2500)
l3 = list("SpatialPolygonsRescale", layout.scale.bar(), offset = c(390000,160000), scale = 5000, fill=c("transparent","black"))
l4 = list("sp.text", c(390000,161000), "0")
l5 = list("sp.text", c(394000,161000), "5000 m")


# Semivariogram for predicted probabilities in the model
modelresults$probs <- (modelresults$lmodel2_pred) * 1
PredMod<- variogram(probs~1, modelresults,boundaries = seq(0,20000, l=51))


# The output and plot of the observed variogram
PredMod
plot(PredMod)
PredMod.vg.mod= fit.variogram(PredMod, vgm(c("Gau", "Exp","Sph")))
PredMod.vg.mod
plot(PredMod, model=PredMod.vg.mod)




mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey, args = list(key = args), corner = c(0.05,.75))
SpatialAuto.pred <- krige(probs~1, modelresults, clip_grid, model=PredMod.vg.mod, nmax = 3)
spplot(SpatialAuto.pred["var1.pred"], sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE), colorkey = list(space = "right", height = 1.0), at = seq(0, 1, .01))+layer(sp.polygons(shp, lwd = 1.5))
summary(SpatialAuto.pred)


# Export output to raster using the rgdal
writeGDAL(SpatialAuto.pred["var1.pred"], "pred.SpatialLogNew.tif")


# Export variance to raster using the rgdal
writeGDAL(SpatialAuto.pred["var1.var"], "variance.SpatialLogNew.tif")


# Semivariogram for autocovariate variable in the model
modelresults$autocov <- (modelresults$autocovariate) * 1
autocovMod<- variogram(autocov~1, modelresults,boundaries = seq(0,20000, l=51))


# The output and plot of the observed variogram
autocovMod
plot(autocovMod)
autocovMod.vg.mod= fit.variogram(autocovMod, vgm(c("Gau", "Exp","Sph")))
autocovMod.vg.mod
plot(autocovMod, model=autocovMod.vg.mod)


# Predicting the probabilities ###
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey, args = list(key = args), corner = c(0.05,.75))
autocov.pred <- krige(autocov~1, modelresults, clip_grid, model=autocovMod.vg.mod, nmax = 3)
spplot(autocov.pred["var1.pred"], sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE), colorkey = list(space = "right", height = 1.0), at = seq(0, 1, .01))+layer(sp.polygons(shp, lwd = 1.5))
summary(autocov.pred)


# Export output to raster using the rgdal
writeGDAL(autocov.pred["var1.pred"], "pred.autocovNew.tif")


# Export variance to raster using the rgdal
writeGDAL(autocov.pred["var1.var"], "variance.autocovNew.tif")


# Semivariogram for residuals in the model
modelresults$residualterm <- (modelresults$res)
residualtermMod<- variogram(residualterm~1, modelresults,boundaries = seq(0,20000,l=51))


# The output and plot of the observed variogram
residualtermMod
plot(residualtermMod)
residualtermMod.vg.mod= fit.variogram(residualtermMod, vgm(c("Gau", "Exp","Sph")))
residualtermMod.vg.mod
plot(residualtermMod, model=residualtermMod.vg.mod)


# Ordinary kriging of the model residuals
mapcolor <-colorRampPalette(brewer.pal(9, "YlOrRd")) (100)
legendArgs <- list(fun = draw.colorkey, args = list(key = args), corner = c(0.05,.75))
residualterm.pred <- krige(residualterm~1, modelresults, clip_grid,model=residualtermMod.vg.mod, nmax = 5), spplot(residualterm.pred["var1.pred"], sp.layout=list(l2,l3,l4,l5),col.regions=mapcolor,scales=list(draw=FALSE), colorkey = list(space = "right", height = 1.0), at = seq(0, 1, .01))+ layer(sp.polygons(shp, lwd = 1.5))
summary(residualterm.pred)


# Export output to raster using the rgdal
writeGDAL(residualterm.pred["var1.pred"], "pred.residualterm.tif")


# Export variance to raster using the rgdal
writeGDAL(residualterm.pred["var1.var"], "variance.residualterm.tif")