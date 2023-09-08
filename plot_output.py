import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import geopandas as gpd
import matplotlib.patches as mpatches
import pandas as pd

# bathymetry
fname = '/Users/jeanmensa/_Sync/Tanzania_data/shapefiles/GEBCO_2014_contours.shp'
shape_bathymetry = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='silver',linewidth=0.5)

# coastline
fname = '/Users/jeanmensa/_Sync/Tanzania_data/shapefiles/coastlines-split-4326/poly_Africa.shp'
shape_coastline = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor=(0.7,0.7,0.7), edgecolor=(0.3,0.3,0.3), linewidth=0.5)

# mangroves
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/Global Mangrove Distribution (USGS)/data/commondata/data0/usgs_mangroves2_TZ.shp'
shape_mangroves = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='green', edgecolor='green', linewidth=0.2)
#mangrove_label = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, facecolor='green', edgecolor='green' )

# coral reefs
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/East_Africa/Tanzania_GCRMN.shp'
shape_coralreefs = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor=[(0,191/255,1)], edgecolor=[(0,140/255,1)], linewidth=0.2)
#coral_label = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, facecolor=[(0,191/255,1)], edgecolor=[(0,140/255,1)])

# villages
#fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/hotosm_tz_villages_Pemba.shp'
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Admin/ZNZ/places/places.shp'
villages = gpd.read_file(fname)

#villages_Z3=['Wesha', 'Tundaua', 'Wambaa', 'Mkoani'] # Misali
villages_Z3=['Bumbwini', 'Mkokotoni', 'Kilindi', 'Nungwi','Gomani'] #TUMCA

# Misali

label = 'Misali'

features = ['HC','SG','RB','Depth']
color_features = {'HC':'blue','SG':'green','RB':'gray','Depth':'pink'}
coeff = {'HC':200,'SG':200,'RB':200,'Depth':2}

xmin = 39.4
ymin = -5.5
xmax = 39.7
ymax = -5.0

# Tumca
label = 'TUMCA'

features = ['HC','Soft coral', 'Seagrass','Rubble','Depth']
color_features = {'HC':'darkblue','Soft coral':'blue','Seagrass':'green','Rubble':'gray','Depth':'pink'}
coeff = {'HC':100,'Soft coral':100,'Seagrass':100,'Rubble':100,'Depth':1}

xmin = 39.11
ymin = -6.05
xmax = 39.33
ymax = -5.7

xmin = 39.12
ymin = -6.02
xmax = 39.31
ymax = -5.74

extent = [xmin,xmax,ymin,ymax]

''' plottint '''

# marxan grid from cost function
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/MarxanData/TUMCA/cost_function.csv'
df = pd.read_csv(fname)
marxan_grid = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
marxan_grid.crs = 'epsg:21037'
marxan_grid = marxan_grid.to_crs('epsg:4326')

for BLM in ['0','0.001','0.01','0.1','1','10','100','1000','10000']: # ['0.001','1','10000']:
 # load output
 fname =  '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/MarxanData/'+label+'/BLM'+BLM+'/output/output_best.csv'
 best = pd.read_csv(fname)

 fname =  '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/MarxanData/'+label+'/BLM'+BLM+'/output/output_ssoln.csv'
 ssoln = pd.read_csv(fname)

 output = marxan_grid.merge(best,left_on='labels',right_on='PUID')
 output = output.merge(ssoln,left_on='labels',right_on='planning_unit')

 fig = plt.figure()
 ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
 ax.set_extent(extent, crs=ccrs.PlateCarree())
 
 ax.add_feature(shape_bathymetry)
 ax.add_feature(shape_coastline)
 
 plt.scatter(x=output.geometry.x, y=output.geometry.y, c=output.SOLUTION.values, s=10, marker='s', transform=ccrs.PlateCarree(), zorder=1, label='BLM '+BLM, cmap='Reds')
 plt.legend()
 #plt.legend([coral_label, manngrove_lavel],['coral areas','mangrove areas'])
 plt.savefig('img/'+label+'/best_solution_BLM'+BLM+'.png',dpi=300,bbox_inches='tight')
 plt.close()
 print('img/'+label+'/best_solution_BLM'+BLM+'.png')

 fig = plt.figure()
 ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
 ax.set_extent(extent, crs=ccrs.PlateCarree())

 ax.add_feature(shape_bathymetry)
 ax.add_feature(shape_coastline)

 plt.scatter(x=output.geometry.x, y=output.geometry.y, c=output.number.values, s=10, marker='s', transform=ccrs.PlateCarree(), zorder=1, label='BLM '+BLM, cmap='PuRd')
 plt.legend()
 #plt.legend([coral_label, manngrove_lavel],['coral areas','mangrove areas'])
 plt.savefig('img/'+label+'/freq_solution_BLM'+BLM+'.png',dpi=300,bbox_inches='tight')
 plt.close()
 print('img/'+label+'/freq_solution_BLM'+BLM+'.png')

