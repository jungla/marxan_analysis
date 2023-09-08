import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import geopandas as gpd
import pandas as pd
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import rasterio
from rasterio.enums import MergeAlg
from functools import partial
from geocube.rasterize import rasterize_image
from geocube.api.core import make_geocube
import rioxarray
 
# bathymetry
fname = '/Users/jeanmensa/_Sync/Tanzania_data/shapefiles/GEBCO_2014_contours.shp'
shape_bathymetry = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='silver',linewidth=0.5)

# coastline
fname = '/Users/jeanmensa/_Sync/Tanzania_data/shapefiles/coastlines-split-4326/poly_Africa.shp'
shape_coastline = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor=(0.7,0.7,0.7), edgecolor=(0.3,0.3,0.3), linewidth=0.5)

# mangroves
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/Global Mangrove Distribution (USGS)/data/commondata/data0/usgs_mangroves2_TZ.shp'
shape_mangroves = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='green', edgecolor='green', linewidth=0.2)
##mangrove_label = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, facecolor='green', edgecolor='green' )

# coral reefs
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/East_Africa/Tanzania_GCRMN.shp'
shape_coralreefs = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor=[(0,191/255,1)], edgecolor=[(0,140/255,1)], linewidth=0.2)
##coral_label = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, facecolor=[(0,191/255,1)], edgecolor=[(0,140/255,1)])

# villages
#fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/hotosm_tz_villages_Pemba.shp'
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Admin/ZNZ/places/places.shp'
villages = gpd.read_file(fname)

#villages_Z3=['Wesha', 'Tundaua', 'Wambaa', 'Mkoani'] # Misali
villages_Z3=['Bumbwini', 'Mkokotoni', 'Gomani'] #TUMCA

label = 'Misali'

xmin = 39.4
ymin = -5.5
xmax = 39.7
ymax = -5.0

label = 'TUMCA'

xmin = 39.11
ymin = -6.05
xmax = 39.33
ymax = -5.7

xmin = 39.12
ymin = -6.02
xmax = 39.31
ymax = -5.74

extent = [xmin,xmax,ymin,ymax]


''' reads cost function from file generated in make_cost_function.py '''

#cost_f = rioxarray.open_rasterio('./MarxanData/'+label+'/fishing_effort.nc').to_dataframe().reset_index()
#cost_f = gpd.GeoDataFrame(cost_f, geometry=gpd.points_from_xy(cost_f.x, cost_f.y))
#cost_f.crs = "EPSG:21037"
#cost_f = cost_f.to_crs('epsg:4326')

vars = [
'cost_function',
'effort_Jarife_Ka',
'effort_Jarife_Ku',
'effort_Jarife',
'effort_Madema_Ka',
'effort_Madema_Ku',
'effort_Madema',
'effort_Mishipi_Ka',
'effort_Mishipi_Ku',
'effort_Mishipi',
'effort_Mtando_Ka',
'effort_Mtando_Ku',
'effort_Mtando',
'effort_Pweza_Ka',
'effort_Pweza_Ku',
'effort_Pweza',
'effort_Juya_Ka',
'effort_Juya_Ku',
'effort_Juya'
]

for var in vars:
 df = pd.read_csv('./MarxanData/'+label+'/'+var+'.csv')
 cost_f = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
 cost_f.crs = 'epsg:21037'
 cost_f = cost_f.to_crs('epsg:4326')
 cost_f = cost_f.loc[cost_f['effort']>0,:]
 
 ''' plotting '''
 
 fig = plt.figure()
 ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
 ax.set_extent(extent, crs=ccrs.PlateCarree())
 
 ax.add_feature(shape_bathymetry,zorder=-1)
 ax.add_feature(shape_coastline,zorder=1)
 ax.add_feature(shape_mangroves,zorder=2)
 #ax.add_feature(shape_coralreefs,zorder=1)

 for z3 in villages_Z3:
  village = villages[villages.name == z3]
  plt.text(float(village.geometry.x)+0.001, float(village.geometry.y)+0.001, z3, fontsize=6, path_effects=[pe.withStroke(linewidth=1, foreground="white")], transform=ccrs.PlateCarree())
  plt.scatter(x=village.geometry.x, y=village.geometry.y, color='gray', s=12, edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree())
 
# plt.tricontourf(cost_f.geometry.x, cost_f.geometry.y, cost_f['effort'].values, 30, transform=ccrs.PlateCarree(), zorder=0)
 plt.scatter(x=cost_f.geometry.x, y=cost_f.geometry.y, c=cost_f['effort'].values, s=10, marker='s', transform=ccrs.PlateCarree(),zorder=0)
 
 #plt.scatter(x=lons, y=lats, color='red', s=effort*10, alpha=1, zorder=100, transform=ccrs.PlateCarree())
 #plt.scatter(x=data.Longitude,y=data.Latitude, color='green', s=extr_effort*10, marker='D', edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree(), label='Fishing effort')
 #plt.colorbar()
 plt.savefig('img/'+label+'/'+var+'.png',dpi=300,bbox_inches='tight')
 print('img/'+label+'/'+var+'.png')
 plt.close()
 
