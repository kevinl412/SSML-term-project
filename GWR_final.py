#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script is to run the GWR model for each year. Initially, the coding part was done in Jupyter notebook but it kept 
crashing due to limited computational power or server capacity. Therefore, only for the GWR part this seperate file is created. 
"""
# Import packages
import rasterio
import geopandas as gpd
import os
import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import Point
from rasterio.features import rasterize


# landuse data preparation
# Directory to the landuse data for North-Holland
landuse_nh_path = '/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML vector data/processed_data/Noord-Holland_bbg2017.gpkg' 

# Load the vector land use file into a GeoDataFrame
landuse_gdf = gpd.read_file(landuse_nh_path)

# Define new categories to have a grouped level of emission 
# Add a numeric column to the data so that can be used as the values of the to-be-made raster cells. 
# This column is called "emision" and gives a value between 1 and 10 to the different land use categories in the degree of emissions, 
# with 10 indicating overall most pollution.

emision_levels = {
    'Spoorterrein': 10,                  # High pollution emissions from train operations
    'Hoofdweg': 10,                      # High pollution emissions from heavy traffic on main roads
    'Vliegveld': 10,                     # Very high pollution emissions from aircraft operations
    'Bebouwd exclusief bedrijfsterrein': 8,   # Moderate to high pollution emissions from urban areas
    'Bedrijfsterrein': 8,                # Moderate to high pollution emissions from industrial areas
    'Semi-bebouwd': 5,                   # Moderate pollution emissions from suburban areas
    'Recreatie': 5,                      # Moderate pollution emissions from recreational areas
    'Glastuinbouw': 7,                   # Moderate pollution emissions from greenhouse farming
    'Landbouw en overig agrarisch': 3,   # Low pollution emissions from agricultural activities
    'Bos': 2,                            # Low pollution emissions from forested areas
    'Droog natuurlijk terrein': 2,       # Low pollution emissions from dry natural areas
    'Nat natuurlijk terrein': 2,         # Low pollution emissions from wetlands
    'Water': 1                           # Very low pollution emissions from water bodies
}

# Create a new column in the GeoDataFrame to store the numerical emission values
landuse_gdf['emision'] = landuse_gdf['bodemgebruik'].map(emision_levels)

# Save the updated GeoDataFrame back to the GeoPackage file, overwriting the original file
landuse_gdf.to_file(landuse_nh_path, driver='GPKG')

print("GeoPackage file updated successfully.")

raw_data_folder_path = '/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML vector data/'
target_data_path = '/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML raster data/processed_data/Noord-Holland_mgr_tot_2020_v2_13092023.tif'

# Create an output directory for the resampled data
output_directory = os.path.join(raw_data_folder_path, 'resampled_data')
os.makedirs(output_directory, exist_ok=True)

# Open the target raster to get its spatial characteristics
with rasterio.open(target_data_path) as target:
    target_transform = target.transform
    target_width = target.width
    target_height = target.height
    target_crs = target.crs

for file_name in os.listdir(raw_data_folder_path):
    if file_name.endswith(('.gpkg', '.shp')):
        print(f'Working on {file_name}')
        name_without_extension, _ = os.path.splitext(file_name)
        file_path = os.path.join(raw_data_folder_path, file_name)

        # Load the vector data
        vector_data = gpd.read_file(file_path)

        # Ensure the vector data has an 'emision' column
        if 'emision' not in vector_data.columns:
            print(f"Skipping {file_name}: 'emision' column not found.")
            continue

        # Set output raster file path
        output_file_name = os.path.join(output_directory, f'Noord-Holland_{name_without_extension}.tif')

        # Define the output raster's metadata
        meta = {
            'driver': 'GTiff',
            'height': target_height,
            'width': target_width,
            'count': 1,
            'dtype': 'float32',  # Change dtype to float32 to accommodate emission values
            'crs': target_crs,
            'transform': target_transform,
            'nodata': 0  # Assuming 0 as nodata value; adjust as necessary
        }

        # Prepare the shapes and values for rasterization
        shapes_and_values = [(
            geom, value) for geom, value in zip(vector_data.geometry, vector_data['emision'])
        ]

        # Rasterize the vector data
        with rasterio.open(output_file_name, 'w', **meta) as out_raster:
            rasterized_layer = rasterize(
                shapes_and_values,
                out_shape=(target_height, target_width),
                transform=target_transform,
                fill=0,  # fill value for 'nodata'
                all_touched=True,
                dtype='float32'
            )
            out_raster.write(rasterized_layer, 1)  # Writing to band 1

        print(f'Data successfully rasterized and aligned: {output_file_name}')
        
        
# GWR

gwr_directory = "/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML vector data/processed_data"
gwr_output_dir = r"/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML vector data/processed_data/predicted data"

# Define the directory for the raster data
mgr_file = "/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML final data/Noord-Holland_mgr_tot_2020_v2_13092023.tif"
bbg_file = "/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML final data/Noord-Holland_Noord-Holland_bbg2017.tif" # This file comes from the landuse data preparation part
sound_file = "/Users/kevinluo/Documents/ADS/SpatialData2/Term project/INFOMSSML final data/Noord-Holland_Noord-Holland_rivm_20220601_Geluid_lden_wegverkeer_2020.tif"

# Initialize an empty dictionary to store results for each year
gwr_results_dict = {}

# Iterate over the years
years = [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021]

for year in years:
    # Define the file paths for NO2 and CO2 shapefiles
    no2_file = os.path.join(gwr_directory, f'Noord-Holland_NO2_{year}.shp')
    co2_file = os.path.join(gwr_directory, f'Noord-Holland_CO2_{year}.shp') 

    no2_input = gpd.read_file(no2_file)
    co2_input = gpd.read_file(co2_file)
    
    gwr_gdf = no2_input.merge(co2_input[['ai_code', 'emision']], on='ai_code', how='inner', suffixes=['_no2','_co2'])
    
    # Zonal statistics on health, noise and landuse data with the raster data for CO2 and NO2    
    with rasterio.open(mgr_file) as src:
        affine = src.transform
        array = src.read(1)
        mgr_zonal_stats = pd.DataFrame(zonal_stats(gwr_gdf, array, affine=affine))
        mgr_zonal_stats = mgr_zonal_stats.rename(columns = {'mean':'mean_mgr'})
        mgr_zonal_stats = mgr_zonal_stats.drop(columns = ['min', 'max', 'count'])
    
    with rasterio.open(sound_file) as src:
        affine = src.transform
        array = src.read(1)
        sound_zonal_stats = pd.DataFrame(zonal_stats(gwr_gdf, array, affine=affine))
        sound_zonal_stats = sound_zonal_stats.rename(columns = {'mean':'mean_sound'})
        sound_zonal_stats = sound_zonal_stats.drop(columns = ['min', 'max', 'count'])
    
    with rasterio.open(bbg_file) as src:
        affine = src.transform
        array = src.read(1)
        bbg_zonal_stats = pd.DataFrame(zonal_stats(gwr_gdf, array, affine=affine, categorical = True))
        bbg_zonal_stats = bbg_zonal_stats.rename(columns = {1.0:'count_bbg1',
                                                            2.0:'count_bbg2',
                                                            3.0:'count_bbg3',
                                                            4.0:'count_bbg4',
                                                            5.0:'count_bbg5',
                                                            6.0:'count_bbg6',
                                                            7.0:'count_bbg7',
                                                            8.0:'count_bbg8',
                                                            9.0:'count_bbg9',
                                                            10.0:'count_bbg10'})
        
    gwr_gdf = pd.concat([gwr_gdf, mgr_zonal_stats, sound_zonal_stats,  bbg_zonal_stats], axis=1)

    # Add centroid to determine the distance to Schiphol airport for each raster cell
    gwr_gdf['centroid'] = gwr_gdf['geometry'].centroid
    gwr_gdf['X'] = gwr_gdf['centroid'].x
    gwr_gdf['Y'] = gwr_gdf['centroid'].y
    gwr_gdf['distance_schiphol'] = gwr_gdf['centroid'].distance(Point(112509.4747363544, 480192.655034324))
    
    # Define the dependent variable
    gwr_y = gwr_gdf['emision_no2'].values.reshape((-1,1))

    # Define the independent variables
    gwr_X = gwr_gdf[['emision_co2', 'mean_mgr', 'mean_sound', 'distance_schiphol', 'count_bbg1', 'count_bbg2','count_bbg3',
                     'count_bbg5', 'count_bbg7', 'count_bbg8', 'count_bbg10']].values
    
    gwr_X = np.nan_to_num(gwr_X, nan=0)  # Replace NaN values with 0

    gwr_u = gwr_gdf['X']
    gwr_v = gwr_gdf['Y']
    gwr_coord = list(zip(gwr_u, gwr_v))

    # Define optimal bandwidth
    gwr_selector = Sel_BW(gwr_coord, gwr_y, gwr_X)
    gwr_bw = gwr_selector.search()

    gwr_results = GWR(gwr_coord, gwr_y, gwr_X, gwr_bw).fit()

    gwr_resid = gwr_results.resid_response

    gwr_summary = gwr_results.summary()
    
    # Monte Carlo test of spatial variability: 200 iterations
    gwr_p_values_stationairity = gwr_results.spatial_variability(gwr_selector, 200)
    
    # Store results, residuals, and summary statistics in the dictionary
    gwr_results_dict[year] = {
        'results': gwr_results,
        'residuals': gwr_resid,
        'summary': gwr_summary,
        'mean_R2': gwr_results.R2,
        'aic': gwr_results.aic,
        'aicc': gwr_results.aicc,
        'bw': gwr_bw,
        'spatial_stat': gwr_p_values_stationairity # the first value is the p-value for the intercept
    }
    
    gwr_gdf['gwr_resid'] = gwr_results_dict[year]['residuals']
    gwr_gdf['gwr_R2'] = gwr_results.localR2
    
    gwr_gdf = gwr_gdf.drop(columns = ['centroid'])
        
    # Create directory
    shapefile_path = os.path.join(gwr_output_dir, f"GWR_predictions_{str(year)}.shp")

    # Save the GeoDataFrame to a shapefile
    gwr_gdf.to_file(filename = shapefile_path, crs = 28992)
    

    
    
for year in gwr_results_dict:
    spatial_stat = gwr_results_dict[year]['spatial_stat']
    # Now you can use spatial_stat for the current year
    print(f"Spatial statistics for the year {year}: {spatial_stat}")
    
    
    
    
    