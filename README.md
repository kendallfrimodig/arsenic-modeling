![](fig/elevated1geology.png)



# Estimating Risk of Arsenic Contamination in Private Wells

By Kendall Frimodig 
Gaston County, and UNC Charlotte

---





## Table of Contents



- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [File Directory](#file-directory)





___


According to a study by the US Geological Survey, naturally occurring arsenic in groundwater varies regionally due to a combination of climate and geology, with greater concentrations found in certain areas of the US including the Interior Plains[¹](https://www.usgs.gov/publications/factors-affecting-temporal-variability-arsenic-groundwater-used-drinking-water-supply). The concentration of naturally occurring arsenic in ground water can also be affected by factors such as groundwater level changes, pumping stresses, evapotranspiration effects, or perhaps mixing of more oxidizing, lower pH recharge water in wetter months[²](https://cfpub.epa.gov/si/si_public_record_report.cfm?dirEntryId=288404).

A recent study has developed machine-learning algorithms to determine what contributes to higher arsenic concentrations in private wells. Factors included geological bedrock type, soil type, drainage class, land use cover, the presence of orchards, known contamination and abandoned mines within 500-feet of each well[³](https://phys.org/news2022-12-algorithms-arsenic-contamination-private-wells.html).


Specifically for North Carolina, there have been several studies regarding elevated arsenic concentrations in groundwater. Findings include associations with metavolcanic or metavolcaniclastic rocks, and the presence of abandoned mines[⁴](http://h2o.enr.state.nc.us/gwp/Arsenic_Studies.htm) [⁵](https://pubs.usgs.gov/sir/2009/5149/), metamorphosed clastic sedimentary rocks, pH [⁶](https://pubs.usgs.gov/sir/2013/5072/pdf/sir2013-5072.pdf) [⁷](https://www.frontiersin.org/articles/10.3389/feart.2018.00111/full), and well depth [⁶](https://pubs.usgs.gov/sir/2013/5072/pdf/sir2013-5072.pdf) [⁸](https://scholars.duke.edu/display/pub772846).

 Owusu et al. conducted a spatial autologistic regression to estimate arsenic hotspots for Gaston County. The data used were 2011-2017. pH and bedrock were found to be associateed with elevated levels. The model was able to predict 90% of the elevated arsenic wells with 80% accuracy[⁴](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6296218/). However, the data used were not split into training and test subsets, so it remains to be seen how testing results 2018-2022 compare to the predicted arsenic level.


The purpose of this study is to enhance a previous modeling project by Owusu et al. []() The sample size for elevated arsenic was relatively low due to the more recent introduction of mandatory testing upon well construction. The project  replicates the original model with a 60% increase in sample size, and explores additional factors that have been identified to affect arsenic concentrations.

[Gaston Water Risk Map](https://gastonwatermap.org/map/?_gl=1*pkmxvu*_ga*MTEzNjAxMzYwOS4xNjc4OTAxMTE3*_ga_JYNZVNN3RN*MTY4MTM1ODgxNy4zLjAuMTY4MTM1ODgxNy4wLjAuMA..*_ga_2LVQGFZ9WX*MTY4MTM1ODgxNy4xMC4wLjE2ODEzNTg4MTcuMC4wLjA.&_ga=2.178785252.1766973899.1681348676-1136013609.1678901117)


[EnviroScan Map](https://enviroscan-map.renci.org/)


[Arsenic Numbers Behind Map](https://sph.unc.edu/wp-content/uploads/sites/112/2022/12/Gaston_WWC2022.pdf)

[NC Well Database](https://www.sciencedirect.com/science/article/abs/pii/S0048969721065578?via%3Dihub)

After the modeling has been completed, actionable recommendations will be made to the Gaston County Health Department. These recommendations will be based on the model's predictions, and will be used to inform future well construction and maintenance.



















---


## File Directory



<br>

#### Notebooks

[Sample Data Cleaning](./notebooks/cleaning-sample-data.ipynb):
- cleans and explores well sampling data

[Permit Data Cleaning](./notebooks/cleaning-permit-data.ipynb):
- cleans and explores database of all permitted wells

[Data Merge](./notebooks/merge-data.ipynb):
- Joins sampling results with permit data

[GIS](./notebooks/gis-processing.ipynb):
- converts well XY data to GeoJSON
- Retreives polygon data from api
- joins geology data to sampled well points

[Pre-Model](./notebooks/pre-model.ipynb):
- investigates coorelations, engineer's features
- calculates autocovariate values for model

[Post-Model](./notebooks/post-model.ipynb):
- takes predicted probabilities and applies them to unsampled wells
- interpolates predicted aresenic class into surface
- displays model parameter tests (AUC, ROC)

[Visualizations](./notebooks/visualization.ipynb):

<br>



#### Data

**originals**
* [Sampling Data](./data/csv/sampled_wells.xlsx)
* [Permitted Wells](./data/csv/permitted_wells.xlsx)

**cleaned**
* [Cleaned Samples](./data/csv/sampled_wells_cleaned.csv)
* [Cleaned Permits](./data/csv/permitted_wells_cleaned.csv)

**spatial**
* [Geology](./data/gis/polygon/geology.GeoJSON): 
* [Well Sample Points](./data/gis/point/ar_samples_w_geol.GeoJSON): 
* [Well Permit Points](./data/gis/point/all_wells.GeoJSON):

**model output**
* [](./data//.): 
* [](./data//.): 
* [](./data//.): 
* [](./data//.): 


<br>

---

## Methods

**Hypothesis:**


**Data Collection**


**Data Cleaning and EDA**



**Preprocessing and Modeling**




---

## Conclusion

