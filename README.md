![](fig/elevated1geology.png)



# Estimating Risk of Arsenic Contamination in Private Wells



---





## Table of Contents


- [File Directory](#file-directory)
- [Methods](#methods)







___

Arsenic contamination in groundwater is a major public health concern across the United States, and North Carolina is no exception. Compared to other states, naturally occurring arsenic in groundwater is less prevalent; however there are clusters of high levels in the Piedmont and Blue Ridge regions.  In this blog post, we will discuss the potential health impacts of arsenic exposure. factors associated with elevated arsenic levels at both regional and county-wide scales, and assess the spatial distribution of elevated clusters in Gaston County, North Carolina.  

This analysis seeks to assess modeling efforts by Owusu et al. 2017, by comparing the [predicted hotspots](https://gastonwatermap.org/map/?_gl=1*pkmxvu*_ga*MTEzNjAxMzYwOS4xNjc4OTAxMTE3*_ga_JYNZVNN3RN*MTY4MTM1ODgxNy4zLjAuMTY4MTM1ODgxNy4wLjAuMA..*_ga_2LVQGFZ9WX*MTY4MTM1ODgxNy4xMC4wLjE2ODEzNTg4MTcuMC4wLjA.&_ga=2.178785252.1766973899.1681348676-1136013609.1678901117)   with additional data obtained from 2018-2022. 

Additional factors influencing arsenic concentrations in Gaston County, not assessed in the study were analyzed as well. The factors associated with arsenic concentrations at a county level are highly specific as demonstrated by Kim et al. 2011, and a model incorporating higher resolution geologic data and additional covariates investigated in this work could help improve the granularity of arsenic distribution in Gaston County. Based on the correlation analysis, agricultural practices and mining activity could play a key role in the local distribution. These factors will be discussed in more detail. Future work would benefit from an increased sample, validating potential clusters or to rule out outliers.

The primary objective is to determine testing distribution, risk distribution, and combine these two data-sets to outline targeted outreach strategies. The deliverable was a list of unsampled wells, and a priority value for testing. Using grid statistics at a 1km level, unsampled wells with little testing in the surrounding area, and elevated samples nearby were prioritized.


Due to these data quality concerns, the modeling portion of the project was de-emphasized but should be re-visited when Gaston County obtains access to the more robust data. This repository is built to be portable, as the covariate data is accessed through api's and spatially calculated and joined in the notebooks - thus removing the barrier of having to re-work geoprocessing in a GUI.





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

[GIS-Data](./notebooks/gis-get-data.ipynb):
- Retreives polygon data from api

[GIS](./notebooks/gis-processing.ipynb):
- converts well XY data to GeoJSON
- joins geology data to sampled well points

[Geocoding QA](./notebooks/geocoding-qa.ipynb):
- assesses geocoding accuracy of sampled wells lacking XY data

[Correlation Analysis](./notebooks/correlation-analysis.ipynb):
- investigates coorelations, engineer's features

[Wells Needing Testing](./notebooks/wells-needing-testing.ipynb):
- anyalyses unsampled wells, prioritizes testing locations 

[Cross Validation](./notebooks/correlation-analysis.ipynb):
- compares predicted and actual values for arsenic concentration

[Visualizations](./notebooks/visualization.ipynb):
- maps various covariates to arsenic concentrations

<br>



<br>

---

## Methods

#### Data Sets
The data provided by Gaston County was made up of two databases. The permit data contained digitized records of new wells constructed 1989 and onward. This included 8,139 wells with address and coordinate information. The spatial distribution of this data is likely representative of the overall private well distribution. For the testing data, 1,714 samples were available. 618 well samples lacked a permit-id, which is required in order to link the coordinate information from the permit database. Two other sources for this data were identified throughout the project, and will be discussed in the limitation section.

#### Data Cleaning for Permitted Wells:
The dataset was filtered to include only the relevant columns, such as permit id, address, coordinates, depth, and date constructed. Columns were renamed to a format consistent with the sample data. Rows with missing values in critical columns were removed. 110 records lacked address information, and 628 lacked XY coordinates.  The cleaned dataset was then saved to a CSV file.

#### Data Cleaning for Sampled Wells:
The dataset was imported and columns 'Collection Date', 'City', and 'ZipCode' were converted to string format. Columns of interest were selected and renamed for consistency. Missing values and typos in city names were identified and corrected, and leading and trailing spaces in city names were removed. Rows with missing or invalid city values were dropped. The 'zip' column was checked for typos and converted to string format. Invalid zip codes were removed, and the dataset length was checked to verify the number of removed entries. Leading and trailing spaces in the 'add' column were removed. The 'date' column was formatted and converted to a datetime object. A new column, 'year_tested', was created to store the year of the sample collection. Non-detect arsenic values were converted to 0, and new columns 'group', 'group_five', and 'group_mcl' were created to categorize the arsenic values based on specified thresholds. Rows with missing arsenic or pH values were dropped, and the cleaned dataset was saved to a CSV file.

#### Exploratory Data Analysis
Performs an exploratory data analysis for arsenic sample data. It reads the cleaned dataset and formats the date column accordingly. Then, it creates two new dataframes, one for elevated arsenic samples between 2011-2017 and the other for elevated arsenic samples between 2018-2022. Several visualizations are created using the seaborn library to analyze the data: Bar charts showing the number of elevated arsenic samples (>0.001) and (>0.005) by year. Line plots of average arsenic levels by year. A line plot showing arsenic levels by day from 2011-2022. The code also checks for the percentage of elevated samples by year and lists all the values for arsenic concentration in the dataset. It's important to note that there's a value of 0.148, which is far above any other result and may be an error.

#### Data Merging

Responsible for merging two datasets (sampled_wells_cleaned.csv and permitted_wells_cleaned.csv) based on their 'id' column. The datasets contain information about sampled wells and permitted wells, respectively. After merging the datasets, the code block counts the number of valid entries in each column, creates a single address string 'full_add' for geocoding, and checks the number of missing values in the newly created columns. The resulting sample data consists of columns 'id', 'full_add', 'date_tested', 'year_tested', 'X', 'Y', 'ar', 'group', 'group_five', 'group_mcl', and 'ph'.To fill in the missing 'X' and 'Y' coordinates, the Google Maps API is used for geocoding. The code defines a function called 'geocode' that takes a row from the sample data as input and returns the 'X' (longitude) and 'Y' (latitude) coordinates for the address in the 'full_add' column. The function is then applied to the sample data, creating two new columns 'geoX' and 'geoY' with the geocoded coordinates. Finally, the code checks for any missing values in the 'geoX' and 'geoY' columns, assigns a 'sample_id' to each row, and saves the merged dataframe with geocoded coordinates as a new CSV file named "ar_samples_merged_xy.csv".

#### GIS Data Acquisition
Retrieves geospatial data for public water systems, geology, county borders, and mineral resources in North Carolina, converts the data into GeoDataFrames, saves the GeoDataFrames as GeoJSON files, and plots the data.


#### GIS Processing and Merging

Processes GIS data related to arsenic samples, merging various datasets to create a comprehensive view of the relationship between arsenic levels and different environmental factors, such as geology, fault lines, land use, mines, and more. The code reads in multiple geoJSON files, including geological data, fault line data, land use data, and point data for mines and arsenic samples. The code then performs various spatial operations, such as clipping data to a specific county, calculating distances between points, and merging related datasets. It also includes data cleaning and preprocessing steps, such as removing missing values, filtering out certain types of mines, and reclassifying land use types for easier analysis. Finally, the code saves the processed data into new geoJSON files for further analysis or visualization.

#### Geocoding Quality Analysis

This script performs a geocoding quality analysis by comparing the X, Y coordinates of records in a permit database to the coordinates returned by the Google Maps API. It aims to assess the accuracy of the geocoding process. The script first reads the merged dataset and creates two subsets: one with the original X, Y coordinates (ar), and another with the geocoded X, Y coordinates (alt). Both subsets are then converted to geodataframes with the appropriate coordinate systems (EPSG:4269 for the original data and EPSG:4326 for the Google Maps data). Next, both geodataframes are transformed to a common projected coordinate system (UTM 17N, EPSG:32617) to calculate distances between the original and geocoded points. The script then iterates through each row of the data, calculating the distance between the original and geocoded coordinates, and storing it in a new column ('dist'). A histogram of the distances is plotted, and records with distances greater than 100 meters are identified as potential outliers. The script then creates an interactive map using the Folium library to display the original and geocoded points for these outliers. This allows users to visually inspect the addresses and determine the accuracy of the geocoding process.

#### Correlation Analysis, Feature Engineering

This script performs correlation analysis and feature engineering on a dataset to prepare it for a classifier model input. In the feature engineering section, the script simplifies the 'belt2' and 'type' columns by replacing their string values with abbreviations. It then creates a new column 'belt_type' by combining 'belt2' and 'type'. This was intended to reduce the count of covariates in a model, since the ‘geocode’ value of the bedrock data has 12 values, whereas the engineered covariate has 6. In a classifier model reducing the number of features is key if no information is lost, and the bedrock type of particular importance still had its own category in the more broad geology category. In the correlation analysis section, the script creates a new dataframe (cr) containing all the numeric data columns, excluding the non-numeric columns. Any categorical data is converted to ‘dummy’ columns, which are individual binary columns for each category of the original column. A heatmap is plotted using the Seaborn library to visualize the correlation matrix. Included in the correlation analysis are the following: Rock Belt*Rock Type, pH, Distance to closest mine, Count of mines to each sample in 3 buffer distances, Distance to closest crop land, Land use category of well location

#### Cross Validation of Predicted Arsenic Hotspots

The cross-validation was conducted to assess the accuracy of the Owusu et al. (2017) predictive model for arsenic levels in Gaston County. The analysis compared the actual levels of arsenic in the county from 2018 to 2022 to the model's predictions, which were trained on data from 2011 to 2017. 

#### Data Visualization and Mapping
This code is for visualizing different geographic data related to arsenic concentration in water, such as geology, mines, land-use type and cropland. Additionally the predicted arsenic hotspots hosted on the Gaston Water Map are visualized with the true arsenic levels. Splits the arsenic concentration data into three groups based on different threshold levels. It creates three maps with a different combinations of the data layers. The first map visualizes arsenic concentration data, highlighting the areas with different concentrations based on the group threshold levels (0.001, 0.005, and 0.01 ppb). The second map includes hydrology data and exposed mines data layers. The third map adds geology and fault line data layers. The fourth map is created to visualize the predicted arsenic concentrations It also adds the arsenic concentration data as points on top of the prediction layer. A fifth map displays the distribution of mines and cropland with elevated samples.

#### Wells at risk: 

Aims to identify wells that should be tested for arsenic concentration based on their proximity to wells with elevated arsenic levels. It follows the findings of Flanagan et al. (2020), which suggest that targeting wells within 1 km of an elevated arsenic well is a more efficient use of resources. The script removes the sampled wells from the comprehensive permitted wells data. It creates a cell grid which counts the number of wells, samples, and elevated results. These metrics are then joined to unsampled wells. It assesses the number of unsampled wells with local testing rates below specific thresholds, such as 50%, 25%, 10%, and 5%. The script also evaluates the number of unsampled wells that are situated within grid cells containing an elevated sample. For each testing rate threshold, it then calculates the number of unsampled wells that are both in grid cells with elevated samples and below the specified testing rate. This information helps determine the most critical areas where further well testing is needed.

the script assigns a priority level to each unsampled well, considering the testing rate and the presence of an elevated sample in the grid cell. Five priority levels are established:
- 1.)	Wells with a testing rate below 5% and in a grid cell with an elevated sample.
- 2.)	Wells with a testing rate below 10% and in a grid cell with an elevated sample.
- 3.)	Wells with a testing rate below 25% and in a grid cell with an elevated sample.
- 4.)	Wells with a testing rate below 50% and in a grid cell with an elevated sample.
- 5.)	Wells with a testing rate below 5%, regardless of the presence of an elevated sample.

The priority levels help to identify the unsampled wells that should be tested first, prioritizing those in areas with higher risks of arsenic contamination and lower testing rates. The script appends this priority information to the dataset of unsampled wells. 




---

## Conclusion

