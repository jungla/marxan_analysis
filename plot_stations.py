import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import geopandas as gpd
import pandas as pd
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
import numpy as np
import matplotlib.patheffects as pe

# bathymetry
fname = '/Users/jeanmensa/_Sync/Tanzania_data/shapefiles/GEBCO_2014_contours.shp'
shape_bathymetry = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='silver',linewidth=0.6)

# coastline
fname = '/Users/jeanmensa/_Sync/Tanzania_data/shapefiles/coastlines-split-4326/poly_Africa.shp'
shape_coastline = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='silver', edgecolor='gray', linewidth=0.6)

# mangroves
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/Global Mangrove Distribution (USGS)/data/commondata/data0/usgs_mangroves2_TZ.shp'
shape_mangroves = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='green', edgecolor='darkgreen', linewidth=0.4)
#mangrove_label = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, facecolor='green', edgecolor='green' )

# coral reefs
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/East_Africa/Tanzania_GCRMN.shp'
shape_coralreefs = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor=[(0,191/255,1)], edgecolor=[(0,140/255,1)], linewidth=0.2)
#coral_label = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, facecolor=[(0,191/255,1)], edgecolor=[(0,140/255,1)])

# villages
#fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/hotosm_tz_villages_Pemba.shp'
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Admin/ZNZ/places/places.shp'
villages = gpd.read_file(fname)

#villages_Z3=['Wesha', 'Tundaua', 'Wambaa', 'Mkoani']
villages_Z3=['Bumbwini', 'Mkokotoni', 'Kilindi', 'Nungwi','Gomani']

# station coordinates 
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/HB_CR_Survey/TUMCA/CoralNet_percentages_TUMCA.shp'
coordr = gpd.read_file(fname)

# plot stations
# Misali
extent = [39.11,39.33,-5.96,-5.63]

# TUMCA
extent = [39.11,39.33,-6.055,-5.7]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())

ax.add_feature(shape_bathymetry)
ax.add_feature(shape_coastline)
ax.add_feature(shape_mangroves)
#ax.add_feature(shape_coralreefs)

for z3 in villages_Z3:
 village = villages[villages.name == z3]
 plt.text(float(village.geometry.x)+0.001, float(village.geometry.y)+0.001, z3, fontsize=8, path_effects=[pe.withStroke(linewidth=1, foreground="white")], transform=ccrs.PlateCarree())
 plt.scatter(x=village.geometry.x, y=village.geometry.y, color='gray', s=12, edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree())

plt.scatter(x=coordr.geometry.x, y=coordr.geometry.y, color='red', s=8, edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree())
plt.savefig('img/'+label+'/stations.png',dpi=300,bbox_inches='tight')
print('img/'+label+'/stations.png')
plt.close()


