import pandas as pd
import numpy as np


# imports data, convert Permit ID, zip to string
df = pd.read_excel("../data/csv/permitted_wells.xlsx", sheet_name="Very_Editted", converters={"Permit ID": str, "zip": str, "Date Permit Issued": str})

# filters columns
df = df[["Permit ID",
         "Addr_ZIP",
         "City",
         "zip",
         "XCOORD",
         "YCOORD",
         "Depth of Well (Feet)",
         "Date Permit Issued"]]

# renames columns
df.columns = ["id","add_zip","city","zip","X","Y","depth","perm_date"]

# deletes any rows with missing values for BOTH add and X columns
df = df.dropna(subset=["add_zip","X"])

# deletes any rows with missing values for id column
df = df.dropna(subset=["id"])

# drops any rows with missing values for city or zip
df = df.dropna(subset=["city","zip"])

# removes zip code from address column for matching purposes later
df['add'] = [s[:-6] for s in df['add_zip']]

# formats date column, combine year_built column
df['perm_date'] = df['perm_date'].str[:10]
df['perm_date'] = pd.to_datetime(df['perm_date'])
df['year_built'] = df['perm_date'].dt.year

# save the cleaned data to a new file in data folder
df.to_csv("../data/csv/permitted_wells_cleaned.csv", index=False)

# Loading the data
df = pd.read_excel("../data/csv/sampled_wells.xlsx", converters={'Collection Date': str,
                                                        'City': lambda x: str(x.strip()),
                                                        'ZipCode': str})

# Selecting the columns of interest
df = df[["Address",
         "City",
         "State",
         "ZipCode",
         "Well Permit #",
         "Collection Date",
         "Arsenic",
         "pH"]]

# Renaming the columns
df.columns = ["add","city","state","zip","id","date","ar","ph"]

# Checking for missing values
df.isnull().sum()

# Checking for typos in city names
set(df['city'])

# corrects typos for city

li = []

for each in df['city']:

    if each in ['GASTONIA', 'Gastonia']:
        li.append('GASTONIA')

    elif each in ['BELMONT', 'Belmont']:
        li.append('BELMONT')

    elif each in ['DALLAS', 'Dallas']:
        li.append('DALLAS')

    elif each in ['MOUNT HOLLY', 'Mt. Holly','MT HOLLY', 'Mount Holly']:
        li.append('MT HOLLY')

    elif each in ['STANLEY','Stanley', 'STALEY']:
        li.append('STANLEY')

    elif each in ['BESSEMER CITY', 'Bessemer City','Bassemer City']:
        li.append('BESSEMER CITY')

    elif each in ['KINGS MOUNTAIN', 'Kings Mountain','KINGS MOUTAIN','King Mtn 1Kings Mountain']:
        li.append('KINGS MTN')

    elif each in['CHERRYVILLE', 'Cherryville']:
        li.append('CHERRYVILLE')

    elif each in['LINCOLNTON','Lincolnton']:
        li.append('LINCOLNTON')

    elif each in['IRON STATION']:
        li.append('IRON STATION')

    elif each in['ALEXIS']:
        li.append('ALEXIS')

    elif each in['LOWEL','LOWELL']:
        li.append('LOWELL')

    elif each in['CROUSE', 'Crouse']:
        li.append('CROUSE')

    elif each in['CRAMERTON']:
        li.append('CRAMERTON')

    elif each in ['MC ADENVILLE']:
        li.append('MCADENVILLE')

    elif each == 'PINEVILLE':
        li.append('PINEVILLE')
    else:
        li.append("")

df['city'] = li

df['city'].value_counts()

# remove leading and trailing spaces from city names
df['city'] = [s.strip() for s in df['city']]

# drop rows with missing city values
df = df[df['city'] != '']

# create index value for tracking in later merges
df['sample_id'] = df.index
len(df)
# check zip codes for typos (aka anything starting with 29 or not 5 digits)
df['zip'].value_counts()

# convert zip column to string
df['zip'] = df['zip'].astype(str)

# check city and zip for bad values
valid_zips  = ['28006',
                '28012',
                '28016',
                '28021',
                '28032',
                '28033',
                '28034',
                '28052',
                '28053',
                '28054',
                '28055',
                '28056',
                '28077',
                '28086',
                '28092',
                '28098',
                '28101',
                '28120',
                '28164']

# loop through dataframe and if zip is not in valid_zips list, remove row
for index, row in df.iterrows():
    if row['zip'] not in valid_zips:
        df.drop(index, inplace=True)

# check number removed

len(df)

# check for blanks in address

print(len(df[df['add'] == '']))
print(len(df[df['add'] == ' ']))

# trim leading and trailing spaces from address

df['add'] = [s.strip() for s in df['add']]

# Converts long date to short date

df['date'] = df['date'].str[:10]
df['date'] = [x.strip() for x in df['date']]
df['date'] = df['date'].replace(regex=['2/1/2021'], value="2021-02-01")

# converts date to datetime

df['date_tested'] = pd.to_datetime(df['date'])

# creates year column

df['year_tested'] = df['date_tested'].dt.year

# converts non-detect arsenic values to 0

li = []
for each in df['ar']:
    if each in('<0.001','< 0.005', '<0.005', '<0.01'):
        li.append(float(0))
    else:
        li.append(float(each))

df['ar'] = li

# Creates a new column to group arsenic values into 0 and 1, 0 for <0.001 and 1 for >=0.001

li = []
for each in df['ar']:
    if each < 0.001:
        li.append('0')
    else: li.append('1')

df['group'] = li

# Creates a new column to group arsenic values into 0 and 1, 0 for <0.005 and 1 for >=0.005

li = []
for each in df['ar']:
    if each < 0.005:
        li.append('0')
    else: li.append('1')

df['group_five'] = li

# Creates a new column to group arsenic values into 0 and 1, 0 for <0.005 and 1 for >=0.01  (MCL)

li = []
for each in df['ar']:
    if each < 0.01:
        li.append('0')
    else: li.append('1')

df['group_mcl'] = li

# if missing arsenic or ph values, drop the row

df = df.dropna(subset=['ar','ph'])

# saves the cleaned data to a new file in data folder

df.to_csv("../data/csv/sampled_wells_cleaned.csv", index=False)

###################################################################
# Exploratory Data Analysis
###################################################################

df = pd.read_csv('../data/csv/sampled_wells_cleaned.csv', converters={'id': lambda x: str(x.strip()),
                                                                'zip': lambda x: str(x.strip()),
                                                                'city': lambda x: str(x.strip()),
                                                                'add': lambda x: str(x.strip())})

df['date_tested'] = pd.to_datetime(df['date_tested'], format='%Y-%m-%d')

# creates 2 new dataframes, one to check the number of elevated arsenic samples 2011-2017
# and the other to check the number of elevated arsenic samples 2018-2022

new = df[df['year_tested'] > 2017]
og = df[df['year_tested'] < 2018]

print(f''' With a threshold of 0.001 mg/L, there are:
number of elevated arsenic samples 2011-2017')
{len(og[og['group'] == 1])}
/ {len(og)}
number of elevated arsenic samples 2018-2022
{len(new[new['group'] == 1])}
/ {len(new)}
''')

print(f''' With a threshold of 0.005 mg/L, there are:
number of elevated arsenic samples 2011-2017
{len(og[og['group_five'] == 1])}
/ {len(og)}
number of elevated arsenic samples 2018-2022
{len(new[new['group_five'] == 1])}
/ {len(new)}
''')

# Check again for missingness

df[(df['year_tested'] < 2018)].isnull().sum()


# Since testing instruments couldn't detect below 0.005 prior to 2018, there are many samples between 0.001 and 0.005 for 2018-2022 thus making it look like the number of elevated detects has increased. 0.001 was chosen as the cutoff still for modeling purposes since it's the 0.001-0.005 is still of concern to health. If the cutoff was 0.005, the bar chart would show 41 elevated samples 2011-2017 and 19 for 2018-2022 instead
# bar chart of elevated arsenic samples (>0.001) 2011-2022

sns.set_style('darkgrid')
sns.set_context('notebook')
sns.set_palette('Set2')
plt.figure(figsize=(10, 5))
sns.countplot(x='year_tested', data=df[df['group'] == 1], color='blue')
plt.title('Elevated Arsenic Samples (>0.001), by Year')
plt.xlabel('Year')
plt.ylabel('Number of Samples')
plt.ylim(0, 25);

# bar chart of elevated arsenic samples (>0.005) 2011-2022

sns.set_style('darkgrid')
sns.set_context('notebook')
sns.set_palette('Set2')
plt.figure(figsize=(10, 5))
sns.countplot(x='year_tested', data=df[df['group_five'] == 1], color='blue')
plt.title('Elevated Arsenic Samples (>0.005), by Year')
plt.xlabel('Year')
plt.ylabel('Number of Samples')
plt.ylim(0, 25);



df= df.sort_values(by=['date_tested'])
viz = df.set_index('date_tested')
avg_yearly = viz['ar'].resample('Y').mean()

sns.set_style('darkgrid')
sns.set_context('notebook')
sns.set_palette('Set2')

plt.figure(figsize=(10, 5))
sns.lineplot(x=avg_yearly.index.year, y=avg_yearly.values, color='blue')
plt.title('Average Arsenic Levels by Year')
plt.xlabel('Year')
plt.ylabel('Average Arsenic Level')
plt.ylim(0, 0.004)
plt.xticks(np.arange(2011, 2023, 1));

# same as above but with elevated arsenic samples defined as >= 0.005

df= df.sort_values(by=['date_tested'])
viz = df.set_index('date_tested')
li = []
for each in viz['ar']:
    if each < 0.005:
        li.append(float(0))
    else: li.append(each)
viz['ar'] = li

avg_yearly = viz['ar'].resample('Y').mean()
sns.set_style('darkgrid')
sns.set_context('notebook')
sns.set_palette('Set2')
plt.figure(figsize=(10, 5))
sns.lineplot(x=avg_yearly.index.year, y=avg_yearly.values, color='blue')
plt.title('Average Arsenic Levels by Year')
plt.xlabel('Year')
plt.ylabel('Average Arsenic Level')
plt.ylim(0, 0.004)
plt.xticks(np.arange(2011, 2023, 1));

# As you can tell from the chart above and below, there is one sample in 2022 that is heavily skewing the data

# line plot of arsenic samples

sns.set_style('darkgrid')
sns.set_context('notebook')
sns.set_palette('Set2')
plt.figure(figsize=(20, 5))
sns.lineplot(x=df['date_tested'], y=df['ar'], color='blue')
plt.title('Arsenic Levels by Day 2011-2022')
plt.xlabel('Date')
plt.ylabel('Arsenic Level')
plt.ylim(0, 0.15);

# Number of total
df['group'].value_counts()

# Check the percentage of elevated samples by year

viz = df[['year_tested','group_five']]
viz = pd.get_dummies(viz, columns=['group_five'])
viz = viz.groupby(['year_tested']).sum()
viz = pd.DataFrame(viz.reset_index())
viz['total'] = (viz['group_five_0'] + viz['group_five_1'])
viz['group_five_0_pct'] = round(((viz['group_five_0'] / viz['total']) * 100), 2)
viz['group_five_1_pct'] = round(((viz['group_five_1'] / viz['total']) * 100), 2)
viz.set_index('year_tested', inplace=True)
viz = viz[['group_five_1', 'group_five_1_pct', 'total']]
viz.columns = ['> 0.005', '% > 0.005', 'Total']
viz

# Lists all values for arsenis concentration in the dataset

df['ar'].value_counts()

# **The value for 0.148 is far above any other result, and may be an error**

##############################################################################
# Merge PErmit and Sample DataFrame
#############################################################################

import googlemaps

# read in cleaned data sets

df1 = pd.read_csv('../data/csv/sampled_wells_cleaned.csv', converters={'id': lambda x: str(x.strip()),
                                                                'zip': lambda x: str(x.strip()),
                                                                'city': lambda x: str(x.strip()),
                                                                'add': lambda x: str(x.strip())})

df2 = pd.read_csv('../data/csv/permitted_wells_cleaned.csv', converters={'id': lambda x: str(x.strip()),
                                                                'zip': lambda x: str(x.strip()),
                                                                'add_zip': lambda x: str(x.strip()),
                                                                'city': lambda x: str(x.strip()),
                                                                'add': lambda x: str(x.strip())})

# joins df2 to df1 on the following columns.

df2 = df2[['id', 'X', 'Y']]
ar = pd.merge(df1, df2, on='id', how='left')

# checks the number of valid entries in each column

ar.count()

# creates single address string for geocoding

ar['full_add'] = ar['add'] + ', ' + ar['city'] + ', ' + 'NC ' + ar['zip']

# checks the number of missing values in the new columns

ar[['X', 'Y']].isnull().sum()

ar = ar[['id','full_add', 'date_tested', 'year_tested', 'X', 'Y', 'ar', 'group', 'group_five', 'group_mcl', 'ph']]

# 972 samples  will have to be geocoded using the Google Maps API
# Run if not all samples got XY coordinates from permit data
# Comment out if no geocoding is needed

api_key = "AIzaSyD4MWa0YgnE8mvIIxxTqJzMbzqippwbOFs"
gmaps_key = googlemaps.Client(key=api_key)

# geocodes useing full address ('full_add') for the ar dataframe,
# outputs X and Y coordinates into seperate new geoX and geoY columns
# the geocode function will use the googlemaps library and geocode api to geocode the addresses

def geocode(row):
    try:
        result = gmaps_key.geocode(row['full_add'])
        geoX = result[0]['geometry']['location']['lng']
        geoY = result[0]['geometry']['location']['lat']
        return pd.Series([geoX, geoY])
    except:
        return pd.Series([np.nan, np.nan])

# applies the geocode function to the ar dataframe
# the geocode function will create two new columns, geoX and geoY, in the ar dataframe

ar[['geoX', 'geoY']] = ar.apply(geocode, axis=1)

# check the number of missing values in the new columns

ar[['geoX', 'geoY']].isnull().sum()

ar['sample_id'] = ar.index

ar.to_csv("../data/csv/ar_samples_merged_xy.csv", index=False)

##################################################################
# Obtain GIS shapefiles for joining to sample point dataset
##################################################################

# Retrievs data from the web and saves it to local folder (./data/gis/)

import geopandas as gpd
import os
import shapely as shp
import requests
import json
from owslib.wfs import WebFeatureService

# Public Water Systems
# obtain public water system data from NConeMap (crs 32119)

url = 'https://services.nconemap.gov/secure/rest/services/NC1Map_Water_Sewer_2004/MapServer/4/query?outFields=*&where=1%3D1&f=geojson'
pws = requests.get(url).json()

# convert to geodataframe

pws = gpd.GeoDataFrame.from_features(pws, crs=4326)

# check crs

print(pws.crs)

# save pws geojson to file

pws.to_file('../data/gis/polygon/pws.geojson', driver='GeoJSON')
pws.plot()

# Geology
# import geology data from geojson on NC OneMap API
# check column names from geology important for modeling
# values assessible here https://www.nconemap.gov/datasets/nconemap::geology/about
# Put the url in a variable
url = "https://services.nconemap.gov/secure/rest/services/NC1Map_Geological/MapServer/2/query?outFields=*&where=1%3D1&f=geojson"

# Get the data

geol = requests.get(url).json()

# convert to geodataframe

gdf_geol = gpd.GeoDataFrame.from_features(geol)

# set crs

gdf_geol.crs = 4269

gdf_geol.to_file(filename='../data/gis/polygon/geology.geojson', driver='GeoJSON')

gdf_geol.plot()

# County Border
# Read in the county border from NC OneMap API (esri rest service)

url = "https://services1.arcgis.com/YBWrN5qiESVpqi92/arcgis/rest/services/ncgs_state_county_boundary/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"

res = requests.get(url).json()

gdf_cnty = gpd.GeoDataFrame.from_features(res)

# save county geojson to file

gdf_cnty.to_file('../data/gis/polygon/county.geojson', driver='GeoJSON')

gdf_cnty.plot()

# Mineral Resource Data System
# dowload zip from web

url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
r = requests.get(url)
with open("../data/csv/mrds-csv.zip", "wb") as code:
    code.write(r.content)

# unzip file

import zipfile
with zipfile.ZipFile("../data/csv/mrds-csv.zip","r") as zip_ref:
    zip_ref.extractall("../data/csv/")

# read in csv

df = pd.read_csv('../data/csv/mrds.csv')
df = df[df['state'] == "North Carolina"]

#convert to geodataframe

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs=4326)

# save to file

gdf.to_file('../data/gis/point/mines.geojson', driver='GeoJSON')
gdf.plot()

##########################################################################
# Gis Processing. Join data to arsenic Samples
##########################################################################

# read in point data 
df = pd.read_csv("../data/csv/ar_samples_merged_xy.csv")

# read in geo jsons

gdf_geol = gpd.read_file("../data/gis/polygon/geology.geojson")
gdf_faults = gpd.read_file("../data/gis/polygon/geofaults.geojson")
gdf_landuse = gpd.read_file("../data/gis/polygon/landuse.geojson")
gdf_cropland = gpd.read_file("../data/gis/polygon/cropland.geojson")
gdf_cnty = gpd.read_file("../data/gis/polygon/county.geojson")
gdf_mines = gpd.read_file("../data/gis/point/mines.geojson")
pred = gpd.read_file("../data/gis/polygon/predicted-arsenic.geojson")
pws = gpd.read_file("../data/gis/polygon/pws.geojson")

# function to print the name and crs for each geodataframe above

def print_crs():
    print('geology crs:', gdf_geol.crs)
    print('faults crs:', gdf_faults.crs)
    print('landuse crs:', gdf_landuse.crs)
    print('cropland crs:', gdf_cropland.crs)
    print('county crs:', gdf_cnty.crs)
    print('mines crs:', gdf_mines.crs)
    print('sample data crs:', ar.crs)

#  plot a polygon and point set

def basic_map(gdf1, gdf2):
    fig, ax = plt.subplots(figsize=(10,10))
    gdf1.plot(ax=ax, color='none', edgecolor='black')
    gdf2.plot(ax=ax, color='red')
    plt.show()

print_crs()

# The code below is what differs this notebook from the main gis processing notebook. if you had to geocode a portion of the samples then you'll have to project them seperately and then merge them back together. This is what the code below does. Comment out next two code blocks if no geocoding was done.
# Create two point datasets, one for points that got their XY coordinates from the merge with permit data,
# and one for points that got their XY from google geocoding. This is because they likely have different coordinate systems.
# Once they have the same coordinate system, the geoocoded points can be merged with the permit points.

ar = df[df['X'].notnull()]
geocoded = df[df['X'].isnull()]

# convert both to geodataframes

ar = gpd.GeoDataFrame(ar, geometry=gpd.points_from_xy(ar.X, ar.Y), crs = "EPSG:4269")

# specify the coordinate system for google maps output (WGS84)

geocoded = gpd.GeoDataFrame(geocoded, geometry=gpd.points_from_xy(geocoded.geoX, geocoded.geoY), crs="EPSG:4326")

# change to NAD83

geocoded = geocoded.to_crs("EPSG:4269")


# concatenate the two geo dataframes

ar = pd.concat([ar, geocoded], ignore_index=True)


# check the length of the merged dataset

len(ar)

# Filter out gaston county from NC counties

gdf_cnty = gdf_cnty[gdf_cnty['County'] == 'Gaston']
gdf_cnty.to_crs("EPSG:4269", inplace=True)

# check all points within county
# by plotting county border and ar points
# using subplots

fig, ax = plt.subplots(figsize=(10,10))
gdf_cnty.plot(ax=ax, color='white', edgecolor='black')
ar.plot(ax=ax, color='red', markersize=1)

# clip points to county (5 dropped)

gdf_ar = gpd.clip(ar, gdf_cnty)

# check the length of the clipped dataset

len(ar)

# # Clip Polygon Data To Gaston County
# clip the geology data to the county boundary

gdf_geol = gpd.clip(gdf_geol, gdf_cnty)
gdf_geol.plot()

# Read in Gaston County Water Map prediction surface, clip by Public Water System Boundaries
# filter out systems where 'wapcs' = "Gaston"

pws = pws[pws['wapcs'] == 'Gaston']
print(pred.crs)
print(pws.crs)

# set crs to 4269 for all dataframes
pred = pred.to_crs("EPSG:4269")
pws = pws.to_crs("EPSG:4269")
print(pred.crs)
print(pws.crs)

# recode the severity column to a more readable format

pred['pred'] = pred['severity'].replace(['good','warn','danger'],['< 0.005','0.005 - 0.01','> 0.01'])

#clip the pred dataframe to the pws dataframe

pred_clipped = pred.overlay(pws, how='symmetric_difference')

# check results

#pred.plot()
pred_clipped.plot()

# save the clipped data to a geojson file

pred_clipped.to_file("../data/gis/polygon/predicted-arsenic-clipped.geojson", driver='GeoJSON')

# ## Merge bedrock type to wells
# build index of column names for retaining after spatial join

columns = ['id',
            'full_add',
            'date_tested',
            'year_tested',
            'ar',
            'group',
            'group_five',
            'group_mcl',
            'ph',
            'sample_id',
            'geometry']

geol_columns = ['geocode', 'belt2','type']
columns.extend(geol_columns)
columns

# Join the bedrock type, "geocode", from the gdf_geol to each well sample point, from gdf_ar,
# based on the polygon from gdf_geol they lie within

gdf_ar = gpd.sjoin(gdf_ar, gdf_geol, how='left', predicate='within')

# drop unneeded columns

gdf_ar = gdf_ar[columns]

# check the result

#gdf_ar.head()


# ## Read in Mining data, Land Use Data, Faults
# Calculate Distances to Wells, Count in Neighborhood
# Calculate Land Use type for well location
# Calculate distance to cropland
# Sourced from USGS Mineral Resources Data System

# Join the landuse type, from the gdf_landuse to each well sample point
# based on the polygon from gdf_landuse they lie within

gdf_ar = gpd.sjoin(gdf_ar, gdf_landuse, how='left', predicate='within')

# verify the join

gdf_ar.head()

# apply labels for landuse types

li = []
for each in gdf_ar['VALUE']:
    if each in [21,22,23,24]:
        li.append('Developed')
    elif each in [31]:
        li.append('Barren')
    elif each in [41,42,43]:
        li.append('Forest')
    elif each in [51,52]:
        li.append('Shrubland')
    elif each in [71,72,73,74]:
        li.append('Herbaceous')
    elif each in [81]:
        li.append('Pasture')
    elif each in [82]:
        li.append('Crops')
    elif each in [90,95]:
        li.append('Wetlands')
    else:
        li.append('Other')

gdf_ar['landuse'] = li

# drop unneeded columns

gdf_ar = gdf_ar.drop(columns=['index_right', 'VALUE'])

# check the result

gdf_ar['landuse'].value_counts()

# re-assign landuse value to be other if not developed,forest, or pasture

li = []
for each in gdf_ar['landuse']:
    if each in ['Developed','Forest','Pasture']:
        li.append(each)
    else:
        li.append('Other')

gdf_ar['landuse'] = li

# check the result

gdf_ar['landuse'].value_counts()

# **Calculate Mining Denisty Variables in Neighborhood**
# change crs for the mines data and arsenic data to a projected CRS for distance calculations

gdf_mines = gdf_mines.to_crs("EPSG:32617")
gdf_ar = gdf_ar.to_crs("EPSG:32617")

# calculate the number of mines in a 250m radius around each well sample point
gdf_ar['num_mines_250'] = gdf_ar.buffer(250).apply(lambda x: gdf_mines.intersects(x).sum())

# calculate the number of mines in a 500m radius around each well sample point
gdf_ar['num_mines_500'] = gdf_ar.buffer(500).apply(lambda x: gdf_mines.intersects(x).sum())

# calculate the number of mines in a 1 km radius around each well sample point
gdf_ar['num_mines_1k'] = gdf_ar.buffer(1000).apply(lambda x: gdf_mines.intersects(x).sum())

# calculate the number of mines in a 2 km radius around each well sample point
gdf_ar['num_mines_2k'] = gdf_ar.buffer(2000).apply(lambda x: gdf_mines.intersects(x).sum())

# calculate the distance to the nearest mine for each well sample point
gdf_ar['dist_to_mine'] = gdf_ar.apply(lambda x: int(gdf_mines.distance(x.geometry).min()), axis=1)

# check the results
gdf_ar.head()


# Calculate distance to farms

# read in cropland polygon data from geojson
gdf_crops = gpd.read_file("../data/gis/polygon/cropland.geojson")

# set crs
gdf_crops.crs = 4269

# read in the geologic fault data from geojson
gdf_faults = gpd.read_file("../data/gis/polygon/geofaults.geojson")

# set crs
gdf_faults.crs = 32119

# change crs for the crop data and fault data to a projected CRS for distance calculations

gdf_crops = gdf_crops.to_crs("EPSG:32617")
gdf_faults = gdf_faults.to_crs("EPSG:32617")

# calculate distance to NEAREST cropland polygon for each well sample point

gdf_ar['dist_crops'] = gdf_ar.apply(lambda x: int(gdf_crops.distance(x.geometry).min()), axis=1)

# calculate distance to faultline or each well sample point

gdf_ar['dist_fault'] = gdf_ar.geometry.apply(lambda x: int(gdf_faults.distance(x).min())
gdf_ar['dist_crops'].describe()
gdf_ar['dist_fault'].describe()

# Change the crs of ar, mines, and crops back to 4269 for plotting

gdf_ar = gdf_ar.to_crs("EPSG:4269")
gdf_mines = gdf_mines.to_crs("EPSG:4269")
gdf_crops = gdf_crops.to_crs("EPSG:4269")
gdf_mines.isna().sum()

# filter mines to bounds of county + 1 degree buffer to account for any mines that may be outside the county boundary

gdf_mines = gdf_mines[gdf_mines.within(gdf_cnty.geometry.unary_union.buffer(1))]

# checks to see if commod1 is missing, and if so fill the value with the value in commod2 or commod3 if present

gdf_mines['commod1'].fillna(gdf_mines['commod3'], inplace=True)
gdf_mines.isna().sum()

# remove mine if dev_stat = "Prospect"

gdf_mines = gdf_mines[gdf_mines['dev_stat'] != 'Prospect']
gdf_mines[['dep_id', 'geometry','oper_type','prod_size', 'score', 'com_type', 'commod1']]
gdf_mines['oper_type'].value_counts()
gdf_mines['prod_size'].value_counts()
gdf_mines['score'].value_counts()
gdf_mines['com_type'].value_counts()
gdf_mines['commod1'].value_counts()

# save mines gdf as geojson

gdf_mines.to_file('../data/gis/point/mines-w-info.geojson', driver='GeoJSON')

# arsenic is commonly concentrated in sulfide-bearing mineral deposits,
# especially those associated with gold mineralization, and it has a strong affinity
# for pyrite, one of the more ubiquitous minerals in the Earth’s crust.
# It is also concentrated in hydrous iron oxides.

############################################################################################
# Geocoding Quality Analysis
#
# Checks the X,Y of google maps api versuse the XY for records matching in the permit database
##############################################################################################

df = pd.read_csv("../data/csv/ar_samples_merged_xy.csv")

# Create two point datasets for those that have BOTH original X Y and geocoded X Y,
# This is because they likely have different coordinate systems.
# Once they have the same coordinate system, distance between the original XY and geocoded XY points will be analyzed

df = df[df['X'].notnull()]
ar = df[['id','full_add','group','X','Y']]
alt = df[['id','full_add','group','geoX','geoY']]
print(len(ar))
print(len(alt))

# Convert both to geodataframes
ar = gpd.GeoDataFrame(ar, geometry=gpd.points_from_xy(ar.X, ar.Y), crs = "EPSG:4269")

# specify the coordinate system for google maps output (WGS84)
alt = gpd.GeoDataFrame(alt, geometry=gpd.points_from_xy(alt.geoX, alt.geoY), crs="EPSG:4326")

# change to projected coordinate system (UTM 17N)
alt = alt.to_crs("EPSG:32617")
ar = ar.to_crs("EPSG:32617")
print(ar.crs)
print(alt.crs)

# using the 'id' column, iterate through each row and calculate the distance between the original X Y and geocoded X Y

dist = []
for i, row in ar.iterrows():
    dist.append(ar.geometry[i].distance(alt.geometry[i]))
df['dist'] = dist

# Based on the summary below, theres a few bad records throwing off the mean distance between original and geocoded coordinates. easy to see by looking at the quartiles. S well delete the outliers and recalculate the mean.
df['dist'].describe()
plt.hist(df['dist'], bins=100)
plt.show()

# subset the dataframe to only those that have a distance greater than 100 meters
check = df[df['dist'] > 100]
len(check)

# check of these, how many have an original X Y
len(check[check['X'].notnull()])

# so we dont have to use the geocoded XY for any of these that are off.
# but to ensure which source is incorrect well plot on a interactive map that lets you
# select each point to see the address field for both the original and geocoded

# create a folium map

import folium
m = folium.Map(location=[35.3, -81.2], zoom_start=5)

# create a feature group for each dataset

ar_fg = folium.FeatureGroup(name="AR")
alt_fg = folium.FeatureGroup(name="Alt")

# iterate through each row in the dataframe and add a marker to the map

for i, row in check.iterrows():
    ar_fg.add_child(folium.Marker(location=[row['Y'], row['X']], popup=row['full_add'],
                                                icon=folium.Icon(color="blue", icon="crosshairs", prefix='fa')))

    alt_fg.add_child(folium.Marker(location=[row['geoY'], row['geoX']], popup=row['full_add'],
                                                icon=folium.Icon(color="red", icon="crosshairs", prefix='fa')))

# add the feature groups to the map
m.add_child(ar_fg)
m.add_child(alt_fg)

# add a layer control to the map
folium.LayerControl().add_to(m)

# save the map
#m.save("../data/alt/gis/point/ar_samples.html")

###########################################################################
# Correlation analysis
# Calculates binary features for a classifier model input, assesses correlations
# of different Variables
###########################################################################

# load in geojson files
gdf_ar = gpd.read_file('../data/gis/point/ar_samples_w_covariates.geojson')
gdf_cnty = gpd.read_file('../data/gis/polygon/county.geojson')

# function to print the name and crs for each geodataframe above
def print_crs():
    print('county crs:', gdf_cnty.crs)
    print('sample data crs:', gdf_ar.crs)
print_crs()

# # Feature Engineering

# Convert elevated arsenic, bedrock type, geologic belt, well depth into dummy variables for modeling. Check for interaction of belt and rock type to guage need for interaction - newly engineered crosstab variables
# Check category levels reflect original study
gdf_ar['belt2'].value_counts()
gdf_ar['type'].value_counts()

# cross tabulate belt2 and type
pd.crosstab(gdf_ar['belt2'], gdf_ar['type'])

# cross tabulate belt2, type, and geocode to check exclusivity
pd.crosstab([gdf_ar['belt2'], gdf_ar['type']], gdf_ar['geocode'])

# Simplify belt and rock type values
# Simplify belt and rock type values
# for belt2, Charlotte Belt = CB, "Inner Piedmont" = IP, "Kings Mountain Belt" = KM
# for type Intrusive Rocks = IR, Metamorphic Rocks = MR

gdf_ar['belt2'] = gdf_ar['belt2'].replace(['Charlotte Belt', 'Inner Piedmont', 'Kings Mountain Belt'], ['CB', 'IP', 'KM'])
gdf_ar['type'] = gdf_ar['type'].replace(['Intrusive Rocks', 'Metamorphic Rocks'], ['IR', 'MR'])

# Crossing belt with rock type would result in 6 columns, whereas the formation code provides more granularity with 12 columns
# create new variable combining belt and rock type

gdf_ar['belt_type'] = gdf_ar['belt2'] + '_' + gdf_ar['type']
gdf_ar['belt_type'].value_counts()

# create dummy variables for belt_type, depth_cat, and group
#gdf_ar = pd.get_dummies(gdf_ar, columns=['belt_type', 'depth_cat', 'landuse'])
# use line above when well depth data is complete
gdf_ar = pd.get_dummies(gdf_ar, columns=['belt_type', 'landuse'])

# ## Assess Correlation
# create a correlation matrix with group_five as the dependent variable
# and all other numeric data as independent variables
gdf_ar.columns
cr = pd.DataFrame(gdf_ar.drop(['id', 'full_add',  'date_tested', 'sample_id',
       'year_tested', 'belt2','type','geometry','geocode'], axis=1))
cr.columns
len(cr)

# scan data frame and convert boolean columns to binary
for col in cr.columns:
    if cr[col].dtype == 'bool':
        cr[col] = cr[col].astype('int')

plt.figure(figsize=(20,20))
# creating mask
mask = np.triu(np.ones_like(cr.corr()))
sns.heatmap(cr.corr()
            , cmap='coolwarm'
            , annot=True
            , fmt='.2f'
            , vmin=-1
            , vmax=1
            , center=0
            , square=True
            , linewidths=0.5
            , cbar_kws={"shrink": 0.5},
              mask=mask)

# export a csv for modeling
df = pd.DataFrame(gdf_ar.drop(['id', 'full_add',  'date_tested', 'sample_id',
       'year_tested', 'belt2','type','geocode','geometry'], axis=1))

df['X'] = gdf_ar['geometry'].x
df['Y'] = gdf_ar['geometry'].y

# convert
# scan data frame and convert boolean columns to binary

for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype('int')

df.columns
df.isnull().sum()

#save df to csv
df.to_csv('../data/csv/model_input.csv', index=False)

################################################################
# modeling
# Sees if incusion of extra covariates will improve model precision_score
###################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, classification_report, roc_curve)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Precision, AUC, Accuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
from scikeras.wrappers import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier

# # Binary Classification
# Logistic regression

df = pd.read_csv('../data/csv/model_input.csv')

# basic logistic regression
y = df['group_five']
X = df.drop(columns=['group_five', 'ar', 'group', 'group_mcl', 'X', 'Y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

sc = StandardScaler()
Z_train = sc.fit_transform(X_train)
Z_test = sc.transform(X_test)

# instantiate the model
lr = LogisticRegression()

# fit the model
lr.fit(Z_train, y_train)

# generate predictions
preds = lr.predict(Z_test).ravel()

# print additional model metrics
print(classification_report(y_test, preds))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
sc = StandardScaler()
Z_train = sc.fit_transform(X_train)
Z_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(32,
                input_shape=(18,),
                activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[TruePositives(),'accuracy'])

history = model.fit(Z_train, y_train, validation_data=(Z_test, y_test), epochs=30, batch_size=512, verbose=0)

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.legend();

# check the auc plot

y_pred_keras = model.predict(Z_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

####################################################################################
# checks accuracy of predictions for Owusu et al. (2017) model prediction surface. Data
# for 2018-2022 is used against the prediction surface to estiamte correct predictions
# # Cross Validation
# Analysis of the predicted arsenic levels for the county on Gaston Water Map. Compares data to the actual levels of arsenic in the county 2018-2022 since the model was trained on data 2011-2017

# read in samples, predicted arsenic and public water systems polygon data
#pred = gpd.read_file("../data/gis/polygon/predicted-arsenic-clipped.geojson")
pred = gpd.read_file("../data/gis/polygon/predicted-arsenic.geojson")
ar = gpd.read_file("../data/gis/point/ar_samples_w_covariates.geojson")

# check length of data
print(len(ar))

# check crs for both dataframes
print(ar.crs)
print(pred.crs)

# Assessing accuracy of the model based on actual arsenic level
pred = pred.to_crs(ar.crs)
pred['pred'] = pred['severity'].replace(['good','warn','danger'],['< 0.005','0.005 - 0.01','> 0.01'])
pred = pred[["geometry", "pred"]]

# join the pred column to ar dataframe using spatial join

ar = gpd.sjoin(ar, pred, how="left", predicate="within")

# check each point has a prediction
print(ar['pred'].isnull().sum())

# drop rows with no prediction
ar = ar.dropna(subset=['pred'])

# create dummy variable for pred column
df= pd.get_dummies(ar, columns=["pred"])

#Did it at least predict the samples above MCL as being at least above 0.005? aka either 1 for the pred_0.005 - 0.01 or 1 for the pred_> 0.01
#On the same note, were the samples below MCL predicted to be below 0.005? aka either 1 for the pred_0.005 - 0.01 or 1 for the pred_< 0.005

# create pred_group_five, pred_group_mcl columns
# pred_group_five = 1 if pred_0.005 - 0.01 or pred_> 0.01 , else 0
# pred_group_mcl = 1 if pred_> 0.01 , else 0

def pred_group_five(df):
    li = []
    for i in df.index:
        if df['pred_0.005 - 0.01'][i] == 1 or df['pred_> 0.01'][i] == 1:
            li.append(1)
        else:
            li.append(0)
    return li

def pred_group_mcl(df):

    li = []
    for i in df.index:
        if df['pred_> 0.01'][i] == 1:
            li.append(1)
        else:
            li.append(0)
    return li

ar['group_five_pred'] = pred_group_five(df)

ar['group_mcl_pred'] = pred_group_mcl(df)

def concordance(df):
    li = []

    for i in df.index:
        if df['ar'][i] < 0.005 and df['pred_< 0.005'][i] == 1:
            li.append(1)
        elif 0.005 <= df['ar'][i] <= 0.01 and df['pred_0.005 - 0.01'][i] == 1:
            li.append(1)
        elif df['ar'][i] > 0.01 and df['pred_> 0.01'][i] == 1:
            li.append(1)
        else:
            li.append(0)
    return li

ar['correct'] = concordance(df)
ar['correct'].value_counts()

# crosstab the correct column with the pred column
pd.crosstab(ar['correct'], ar['pred'])


# run the concordace again on a subset for years < 2018
ar_2017 = ar[ar['year_tested'] < 2018]
df_2017 = df[df['year_tested'] < 2018]
ar_2017['correct'] = concordance(df_2017)
pd.crosstab(ar_2017['correct'], ar_2017['pred'])


# run the concordace again on a subset for years > 2018
ar_2018 = ar[ar['year_tested'] > 2018]
df_2018 = df[df['year_tested'] > 2018]
ar_2018['correct'] = concordance(df_2018)
pd.crosstab(ar_2018['correct'], ar_2018['pred'])



# Assessing accuracy of the model based on binary classification of arsenic level

# definition to calculate accuracy (tn + tp) / (tp + tn + fp + fn)
# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)
# precision = tp / (tp + fp)
# false positive rate = fp / (fp + tn)
# false negative rate = fn / (fn + tp)

def classifier_metrics(obs, pred):
    A = 0
    B = 0
    C = 0
    D = 0
    for i in range(len(obs)):
        if obs[i] == 1 and pred[i] == 1:
            A += 1
        elif obs[i] == 0 and pred[i] == 1:
            B += 1
        elif obs[i] == 1 and pred[i] == 0:
            C += 1
        elif obs[i] == 0 and pred[i] == 0:
            D += 1

    sensitivity = round(A / (A + C), 2)
    specificity = round(D / (D + B), 2)

    print(f'Confusion Matrix: \n{A} {B} \n{C} {D}')
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Accuracy: {round((A + D) / (A + B + C + D), 2)}')
    print(f'Precision: {round(A / (A + B), 2)}')
    print(f'False Positive Rate: {round(B / (B + D), 2)}')
    print(f'False Negative Rate: {round(C / (C + A), 2)}')

# Model metrics using 5ppb as the threshold for elevated prediction (combine 'warn' and 'danger' from map)
# for all samples
print('Model Accuracy for All Samples, Threshold = 5ppb')
classifier_metrics(ar['group_five'], ar['group_five_pred'])

# for samples < 2018
ar_2017.reset_index(inplace=True)
print('Model Accuracy for Samples Tested Before 2018, Threshold = 5ppb')
classifier_metrics(ar_2017['group_five'], ar_2017['group_five_pred'])
# for samplees > 2018
ar_2018.reset_index(inplace=True)
print('Model Accuracy for Samples Tested After 2018, Threshold = 5ppb')
classifier_metrics(ar_2018['group_five'], ar_2018['group_five_pred'])

# Model metrics using 5ppb as the threshold for elevated prediction ( 'danger' from map = 1)

print('Model Accuracy for All Samples, Threshold = 10ppb')
classifier_metrics(ar['group_mcl'], ar['group_mcl_pred'])

print('Model Accuracy for Samples Tested Before 2018, Threshold = 10ppb')
classifier_metrics(ar_2017['group_mcl'], ar_2017['group_mcl_pred'])

print('Model Accuracy for Samples Tested After 2018, Threshold = 10ppb')
classifier_metrics(ar_2018['group_mcl'], ar_2018['group_mcl_pred'])

ar['group_five'].value_counts()

##################################################################################
# Visualization
# create maps with elevated results, and other covariates visualized
##################################################################################

import esda
import geojson as gs
import splot
from splot.esda import plot_moran
import contextily

# load in geojson files

gdf_ar = gpd.read_file('../data/gis/point/ar_samples_w_covars.geojson')
gdf_cnty = gpd.read_file('../data/gis/polygon/county.geojson')
gdf_geol = gpd.read_file('../data/gis/polygon/geology.geojson')
gdf_openmines = gpd.read_file('../data/gis/polygon/exposedmines.geojson')
gdf_mines = gpd.read_file('../data/gis/point/mines.geojson')
gdf_water = gpd.read_file('../data/gis/polygon/water.geojson')
gdf_faults = gpd.read_file('../data/gis/polygon/geofaults.geojson')
gdf_cropland = gpd.read_file('../data/gis/polygon/cropland.geojson')
gfd_pred = gpd.read_file('../data/gis/polygon/predicted-arsenic-clipped.geojson')

# function to print the name and crs for each geodataframe above
def print_crs():
    print('geology crs:', gdf_geol.crs)
    print('faults crs:', gdf_faults.crs)
    print('county crs:', gdf_cnty.crs)
    print('mines crs:', gdf_mines.crs)
    print('open mines crs:', gdf_openmines.crs)
    print('water crs:', gdf_water.crs)
    print('cropland crs:', gdf_cropland.crs)
    print('sample data crs:', gdf_ar.crs)
    print('predicted data crs:', gfd_pred.crs)

print_crs()

gdf_faults.to_crs(epsg=4269, inplace=True)

# split gdf_ar into 2 for mapping, based on group value (>0.001)
nondetect = gdf_ar[gdf_ar['group'] == 0]
detect = gdf_ar[gdf_ar['group'] == 1]

# splits gdf_ar into 2 for mapping, based on group_five value (>0.005)
nondetect5 = gdf_ar[gdf_ar['group_five'] == 0]
detect5 = gdf_ar[gdf_ar['group_five'] == 1]

# splits gdf_ar into 2 for mapping, based on group_five value (>0.01)
nondetectmcl = gdf_ar[gdf_ar['group_mcl'] == 0]
detectmcl = gdf_ar[gdf_ar['group_mcl'] == 1]

f, ax = plt.subplots(1, figsize=(20, 20), dpi=300)

gdf_cnty.plot(
    color="none",
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)
nondetect.plot(
    color="grey",
    edgecolor="white",
    linewidth=0.0,
    alpha=0.5,
    ax=ax,
)
detect.plot(
    color="yellow",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
detect5.plot(
    color="orange",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
detectmcl.plot(
    color="red",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
contextily.add_basemap(
    ax,
    crs=detect.crs,
    source=contextily.providers.Stamen.Terrain,
)
ax.set_axis_off()

f, ax = plt.subplots(1, figsize=(20, 20), dpi=300)

gdf_cnty.plot(
    color="none",
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)
gdf_water.plot(
    color="blue",
    edgecolor="blue",
    linewidth=0.3,
    ax=ax
)
gdf_openmines.plot(
    color="pink",
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)
nondetect.plot(
    color="grey",
    edgecolor="white",
    linewidth=0.0,
    alpha=0.5,
    ax=ax,
)
detect.plot(
    color="yellow",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
detect5.plot(
    color="orange",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
detectmcl.plot(
    color="red",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
contextily.add_basemap(
    ax,
    crs=detect.crs,
    source=contextily.providers.Stamen.TerrainBackground,
)
ax.set_axis_off()

f, ax = plt.subplots(1, figsize=(20, 20), dpi=300)
nondetect.plot(
    color="grey",
    edgecolor="white",
    linewidth=0.0,
    alpha=0.5,
    ax=ax,
)
detect.plot(
    color="yellow",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
detect5.plot(
    color="orange",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)

detectmcl.plot(
    color="red",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
)
gdf_geol.plot(
    color="none",
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)
gdf_faults.plot(
    color="brown",
    edgecolor="brown",
    linewidth=4,
    ax=ax
)
contextily.add_basemap(
    ax,
    crs=detect.crs,
    source=contextily.providers.Stamen.Terrain,
)

ax.set_axis_off()

# read in polygon data

pred = gpd.read_file("../data/gis/polygon/predicted-arsenic-clipped.geojson")

fig, ax = plt.subplots(figsize=(20,10))
pred.plot(ax=ax, column='pred', cmap='OrRd', legend=True, alpha=0.2,
             legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5)}, categories=['< 0.005','0.005 - 0.01','> 0.01'])
nondetect5.plot(
    color="grey",
    edgecolor="white",
    linewidth=0.0,
    alpha=0.5,
    ax=ax,
    Legend=True
)
detect5.plot(
    color="orange",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
    Legend=True
)
detectmcl.plot(
    color="red",
    edgecolor="black",
    linewidth=0.3,
    alpha=1,
    ax=ax,
    Legend=True
)
ctx.add_basemap(ax, crs=ar.crs.to_string(), source=ctx.providers.Stamen.Terrain)

################################################################################
# Wells needing testing
# Assess percent of wells tested using a grid 
# Creates a narrowed down list of wells that have yet to be tested, but are 
# within a certain range of a well that has been tested. Per the results of 
# flanagan et al (2020) targeteting wells that are within 1.5 km of a well that
#  has shown elevated arsenic levels is a more efficient use of resources than 
# targeting all wells in a larger area as seen on Gaston Water Map. While the 
# water map outlines the broader area where elevated resutls are present, the 
# clustering of arsenic appears to be much more localized within the outlined 
# risk area.



import geopandas as gpd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fiona as fio
import shapely as shp  
import math
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# Read in all wells csv files

wells_all = pd.read_csv('../data/csv/permitted_wells_cleaned.csv')

# convert to geodataframe

wells_all = gpd.GeoDataFrame(wells_all, geometry=gpd.points_from_xy(wells_all.X, wells_all.Y, crs='EPSG:4269'))
wells_ar = gpd.read_file('../data/gis/point/ar_samples_w_covariates.geojson')
cnty = gpd.read_file('../data/gis/polygon/county_gaston.geojson')

print(wells_ar.crs)
print(cnty.crs)


wells_elev = wells_ar[wells_ar['group_five'] == 1]

# plot the two geo dataframes in the same plot same row

fig, ax = plt.subplots(1,2, figsize=(10,5))
wells_all.plot(ax=ax[0], color='blue', markersize=1)
wells_ar.plot(ax=ax[1], color='red', markersize=1)
plt.show()


# convert both geodataframes to projected coordinate system for network analysis

wells_all = wells_all.to_crs('EPSG:32617')
wells_ar = wells_ar.to_crs('EPSG:32617')
cnty = cnty.to_crs('EPSG:32617')

# create a blank polygon grid polygon that's the shape of a county geojson polygon, 
# where each grid is 1 mile by 1 mile

# Function to create a polygon grid
def create_polygon_grid(county_polygon, grid_size):
    xmin, ymin, xmax, ymax = county_polygon.bounds

    rows = int(np.ceil((ymax - ymin) / grid_size))
    cols = int(np.ceil((xmax - xmin) / grid_size))

    grid_polygons = []
    for i in range(cols):
        for j in range(rows):
            x1 = xmin + i * grid_size
            x2 = xmin + (i + 1) * grid_size
            y1 = ymin + j * grid_size
            y2 = ymin + (j + 1) * grid_size
            grid_polygons.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

    return MultiPolygon(grid_polygons)


# Get the county polygon (assuming only one polygon in the GeoJSON file)
county_polygon = cnty.iloc[0].geometry


# Create a 250m grid within the county polygon
grid_size_meters = 1000
grid = create_polygon_grid(county_polygon, grid_size_meters)

# Filter the grid cells that are within the county polygon
grid_within_county = [poly for poly in grid.geoms if poly.intersects(county_polygon)]

# Create a GeoDataFrame from the grid cells and save it to a new GeoJSON file
grid_gdf = gpd.GeoDataFrame({"geometry": grid_within_county}, crs=cnty.crs)
grid_gdf.to_file("../data/gis/polygon/county_grid.geojson", driver="GeoJSON")


len(grid_gdf)

f, ax = plt.subplots(1, figsize=(10, 10))


wells_all.plot(
    color="blue",
    edgecolor="none",
    alpha=0.1,
    markersize=3,
    ax=ax,
)

grid_gdf.plot(
    color="none",
    edgecolor="red",
    linewidth=0.1,
    alpha=0.5,
    ax=ax,
)
cnty.plot(
    color="none",
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)
#contextily.add_basemap(
#    ax,
#    crs=detect.crs,
#    source=contextily.providers.Stamen.Terrain,
#)
ax.set_axis_off()

# for each grid cell, count the number of points from the 'all_wells' geodataframe that are within the grid cell
# and add that count to a new column in the grid geodataframe

grid_gdf['count_wells'] = grid_gdf.apply(lambda row: len(wells_all[wells_all.within(row.geometry)]), axis=1)

# count the number of wells sampled in each grid cell

grid_gdf['count_samples'] = grid_gdf.apply(lambda row: len(wells_ar[wells_ar.within(row.geometry)]), axis=1)

# count whether the grid cell contains any wells that have a 'group_five' value of 1

grid_gdf['elevated'] = grid_gdf.apply(lambda row: len(wells_ar[(wells_ar.within(row.geometry)) & (wells_ar['group_five'] == 1)]), axis=1)

print("Total grid cells: ", len(grid_gdf))
print("Grid cells with wells: ", len(grid_gdf[grid_gdf['count_wells'] > 0]))
print("Grid cells with samples: ", len(grid_gdf[grid_gdf['count_samples'] > 0]))


# calculate the ratio of wells sampled to wells permitted as a percentage in a new column
# first drop any features where there are less than 5 wells permitted

grid_samples = grid_gdf[grid_gdf['count_wells'] > 5]

grid_samples['ratio'] = grid_samples['count_samples'] / grid_samples['count_wells'] * 100

# remove any grid cells where the ratio is GT 100% 
# (i.e. anomolies in geocoding in the grid cell having more samples than permitted wells)
# since this will mess up the color scale of the mapping

grid_samples = grid_samples[grid_samples['ratio'] <= 100]

# clip grid samples to county

grid_samples = gpd.clip(grid_samples, cnty)

print(grid_samples.crs)
print(cnty.crs)
print(wells_elev.crs)

wells_elev = wells_elev.to_crs('EPSG:32617')

# map the grid_samples geodataframe to show the ratio of wells sampled to wells permitted
# using cloropleth map

f, ax = plt.subplots(1, figsize=(10, 10), dpi=300)

grid_samples.plot(
    column="ratio",
    cmap="OrRd_r",
    edgecolor="none",
    alpha=0.5,
    scheme='quantiles',
    ax=ax,
    legend=True
)
cnty.plot(
    color="none",
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

wells_all.plot(
    color="grey",
    edgecolor="none",
    alpha=0.3,
    markersize=1,
    ax=ax,
)

wells_ar.plot(
    color="green",
    edgecolor="none",
    markersize=1,
    ax=ax
)

wells_elev.plot(
    color="red",
    edgecolor="none",
    markersize=3,
    ax=ax
)

#contextily.add_basemap(
#    ax,
#    crs=detect.crs,
#    source=contextily.providers.Stamen.Terrain,
#)
ax.set_axis_off()

# subset the grid geodataframe to only include grid cells with an elevated sample 

grid_elevated = grid_gdf[grid_gdf['elevated'] > 0]

# remove the wells that have been sampled from the all wells dataframe

wells_unsampled = wells_all[~wells_all['id'].isin(wells_ar['id'])]
len(wells_unsampled)

# assign the ratio of wells sampled to wells permitted in the grid cell to each well in the wells_unsampled dataframe

wells_unsampled['ratio'] = wells_unsampled.apply(lambda row: grid_samples[grid_samples.geometry.contains(row.geometry)]['ratio'].values, axis=1)
wells_unsampled['ratio'] = wells_unsampled['ratio'].apply(lambda x: x[0] if len(x) > 0 else None)

# add an indicator in the wells_unsampled dataframe to identify cells within the grid cells with elevated samples

wells_unsampled['elevated'] = wells_unsampled.apply(lambda row: grid_elevated[grid_elevated.geometry.contains(row.geometry)]['elevated'].values, axis=1)
wells_unsampled['elevated'] = wells_unsampled['elevated'].apply(lambda x: x[0] if len(x) > 0 else None)


print(F'''Number of Unsampled wells
    {len(wells_unsampled)}''')

print(F'''Number of Unsampled wells
where the overall testing rate is less than than 50%
    {len(wells_unsampled[wells_unsampled['ratio'] < 50])}''')

print(F'''Number of Unsampled wells
where the overall testing rate is less than than 25%
    {len(wells_unsampled[wells_unsampled['ratio'] < 25])}''')

print(f'''Number of Unsampled wells 
where the overall testing rate is less than 10%
    {len(wells_unsampled[wells_unsampled['ratio'] < 10])}''')

print(f'''Number of Unsampled wells 
where the overall testing rate is less than 5%
    {len(wells_unsampled[wells_unsampled['ratio'] < 5])}''')

print(f'''Number of Unsampled wells 
where the overall testing rate is 0
    {len(wells_unsampled[wells_unsampled['ratio'] < 5])}''')

# check the length of the wells_unsampled dataframe to make sure it's less than the original dataframe

print(f'''Number of unsampled wells in a cell with an elevated sample
    {len(wells_unsampled[(wells_unsampled['elevated'] == 1)])}''')

print(f'''Number of Unsampled wells in a cell with an elevated sample, 
where the overall testing rate is less than 50%
    {len(wells_unsampled[(wells_unsampled['ratio'] < 50) & (wells_unsampled['elevated'] == 1)])}''')

print(f'''Number of Unsampled wells in a cell with an elevated sample, 
where the overall testing rate is less than 25%
    {len(wells_unsampled[(wells_unsampled['ratio'] < 25) & (wells_unsampled['elevated'] == 1)])}''')

print(f'''Number of Unsampled wells in a cell with an elevated sample, 
where the overall testing rate is less than 10%
    {len(wells_unsampled[(wells_unsampled['ratio'] < 10) & (wells_unsampled['elevated'] == 1)])}''')

print(f'''Number of Unsampled wells in a cell with an elevated sample, 
where the overall testing rate is less than 5%
    {len(wells_unsampled[(wells_unsampled['ratio'] < 5) & (wells_unsampled['elevated'] == 1)])}''')

# create a loop where a priority number is set to each unsampled well, based on the ratio of wells sampled to wells permitted
# and the presence of an elevated sample in the grid cell

li = []

for index, row in wells_unsampled.iterrows():
    priority = 0
    if row['ratio'] < 5 and row['elevated'] == 1:
        priority = 1
    elif row['ratio'] < 10 and row['elevated'] == 1:
        priority = 2
    elif row['ratio'] < 25 and row['elevated'] == 1:
        priority = 3
    elif row['ratio'] < 50 and row['elevated'] == 1:
        priority = 4
    elif row['ratio'] < 5:
        priority = 5

    li.append(priority)

wells_unsampled['priority'] = li

wells_unsampled['priority'].value_counts()

# write to a csv file

wells_unsampled.to_csv('../data/csv/wells_needing_testing.csv', index=False)