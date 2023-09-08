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
import rioxarray
import scipy

# bathymetry
#fname = '/Users/jeanmensa/_Sync/Tanzania_data/shapefiles/GEBCO_2014_contours.shp'
#shape_bathymetry = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='silver',linewidth=0.6)

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

# MPA
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/MCA/TUMCA.shp'
shape_MPA = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor=[(0,191/255,1)], edgecolor=[(0,140/255,1)], linewidth=0.2)

# features layers 
# fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Pemba/coral_reef/CoralNet_Coordinates_24may22.shp' # Misali
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/HB_CR_Survey/TUMCA/CoralNet_percentages_TUMCA.shp'
data = gpd.read_file(fname)

# deep stations to fill gaps in stations > 50, this should be taken from the original survey design
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/HB_CR_Survey/TUMCA/Deep_coords_TUMCA.shp'
coords = gpd.read_file(fname)
coords = coords.to_crs('EPSG:4326')
#coords = coords[coords['Area'] == 'TUMCA']

data = pd.concat([coords, data])
data = data.drop(['id','Area','Area_ID','UID'],axis=1)

data.loc[data['Row Labels'] != data['Row Labels'],'Sand'] = 1 # fill deep stations with Sand
data = data.fillna(0) # fill everything else with 0

data['Depth'] = abs(data['Depth'].astype('float')) # set positive depth values
data['Labels'] = data.index+1 # create label field for Marxan

# villages
#fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/GIS/hotosm_tz_villages_Pemba.shp'
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Admin/ZNZ/places/places.shp'
villages = gpd.read_file(fname)

#villages_Z3=['Wesha', 'Tundaua', 'Wambaa', 'Mkoani'] # Misali
villages_Z3=['Bumbwini', 'Mkokotoni', 'Kilindi', 'Nungwi','Gomani'] #TUMCA
villages_Z3=['Bumbwini', 'Mkokotoni', 'Gomani'] #TUMCA


# grid final, features are to be projected on this grid

fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/grids/TUMCA/marxan_grid_TUMCA_500m.shp'
marxan_grid = gpd.read_file(fname)

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

# interpolation to 500m grid

ilons = marxan_grid.geometry.x
ilats = marxan_grid.geometry.y

data = data.to_crs('epsg:21037')

lons = data.geometry.x
lats = data.geometry.y

data_i=data.iloc[:0]
data_i['geometry'] = marxan_grid['geometry']

for var in features:
 var_i = griddata((lons,lats), data[var], (ilons, ilats), method='linear')
 data_i[var] = var_i


for var in features:
 fig = plt.figure()
 ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
 ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(shape_bathymetry)
 ax.add_feature(shape_coastline)
 ax.add_feature(shape_mangroves)
# ax.add_feature(shape_coralreefs)

 for z3 in villages_Z3:
  village = villages[villages.name == z3]
  plt.text(float(village.geometry.x)+0.001, float(village.geometry.y)+0.001, z3, fontsize=8, path_effects=[pe.withStroke(linewidth=1, foreground="white")], transform=ccrs.PlateCarree())
  plt.scatter(x=village.geometry.x, y=village.geometry.y, color='gray', s=12, edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree())

 data_n = data_i.dropna(subset=var)
 data_n = data_n.to_crs('epsg:4326')

# plt.tricontourf(data_n.geometry.x, data_n.geometry.y, data_n[var].values, 30, transform=ccrs.PlateCarree(),zorder=0)
 plt.scatter(x=data_n.geometry.x, y=data_n.geometry.y, color=color_features[var], s=data_n[var].astype('float')*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree())
 #plt.scatter(x=data.geometry.x, y=data.geometry.y, color=color_features[var], s=data[var].astype('float')*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree())

 plt.scatter(x=-99, y=-99, color=color_features[var], s=0.6*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=0, transform=ccrs.PlateCarree(), label='60%')
 plt.scatter(x=-99, y=-99, color=color_features[var], s=0.5*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=0, transform=ccrs.PlateCarree(), label='50%')
 plt.scatter(x=-99, y=-99, color=color_features[var], s=0.4*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=0, transform=ccrs.PlateCarree(), label='40%')
 plt.scatter(x=-99, y=-99, color=color_features[var], s=0.3*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=0, transform=ccrs.PlateCarree(), label='30%')
 plt.scatter(x=-99, y=-99, color=color_features[var], s=0.2*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=0, transform=ccrs.PlateCarree(), label='20%')
 plt.scatter(x=-99, y=-99, color=color_features[var], s=0.1*coeff[var], edgecolor='black', linewidths=0.5, alpha=1, zorder=0, transform=ccrs.PlateCarree(), label='10%')
 plt.legend(loc = 'lower right', title=var)
 #plt.legend([coral_label, manngrove_lavel],['coral areas','mangrove areas'])
 plt.savefig('img/'+label+'/features_'+var+'_'+label+'.png',dpi=300,bbox_inches='tight')
 print('img/'+label+'/features_'+var+'_'+label+'.png')
 plt.close()


''' features from FPM '''

gears = ['Jarife','Juya','Madema','Mishipi','Mtando','Pweza']

# attributes 'f_KuN', 'f_KaN', 'Mammals', 'Sea turtles', 'Sharks', 'Rays', 'Permanent', 'Temporary', 'None'

# prepare for interpolation

f_S_t = np.zeros(len(data_i.geometry.y))
f_R_t = np.zeros(len(data_i.geometry.y))
f_T_t = np.zeros(len(data_i.geometry.y))
f_M_t = np.zeros(len(data_i.geometry.y))

for g in range(len(gears)):
 f_gear = rioxarray.open_rasterio('/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/FPM/'+gears[g]+'.nc')

 interp = scipy.interpolate.RegularGridInterpolator((f_gear.y.values, f_gear.x.values), np.squeeze(f_gear['Sharks'].values), bounds_error=False, fill_value=0, method='linear')
 f_S = interp((data_i.geometry.y, data_i.geometry.x))

 interp = scipy.interpolate.RegularGridInterpolator((f_gear.y.values, f_gear.x.values), np.squeeze(f_gear['Rays'].values), bounds_error=False, fill_value=0, method='linear')
 f_R = interp((data_i.geometry.y, data_i.geometry.x))

 interp = scipy.interpolate.RegularGridInterpolator((f_gear.y.values, f_gear.x.values), np.squeeze(f_gear['Sea turtles'].values), bounds_error=False, fill_value=0, method='linear')
 f_T = interp((data_i.geometry.y, data_i.geometry.x))

 interp = scipy.interpolate.RegularGridInterpolator((f_gear.y.values, f_gear.x.values), np.squeeze(f_gear['Mammals'].values), bounds_error=False, fill_value=0, method='linear')
 f_M = interp((data_i.geometry.y, data_i.geometry.x))

 f_S[f_S != f_S] = 0
 f_R[f_R != f_R] = 0
 f_T[f_T != f_T] = 0
 f_M[f_M != f_M] = 0

 f_S_t = f_S_t + f_S/len(gears)
 f_R_t = f_R_t + f_R/len(gears)
 f_T_t = f_T_t + f_T/len(gears)
 f_M_t = f_M_t + f_M/len(gears)

def scale(var):
 return (var-np.min(var))/np.max(var-np.min(var))

data_i['Sharks'] = scale(f_S_t)
data_i['Rays'] = scale(f_R_t)
data_i['Turtles'] = scale(f_T_t)
data_i['Mammals'] = scale(f_M_t)

data_i['Sharks'] = f_S_t
data_i['Rays'] = f_R_t
data_i['Turtles'] = f_T_t
data_i['Mammals'] = f_M_t

species_l = ['Sharks','Rays','Turtles','Mammals']

data_i = data_i.to_crs('epsg:4326')

for var in species_l:
 fig = plt.figure()
 ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
 ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(shape_bathymetry)
 ax.add_feature(shape_coastline)
 ax.add_feature(shape_mangroves)
# ax.add_feature(shape_coralreefs)

 for z3 in villages_Z3:
  village = villages[villages.name == z3]
  plt.text(float(village.geometry.x)+0.001, float(village.geometry.y)+0.001, z3, fontsize=6, path_effects=[pe.withStroke(linewidth=1, foreground="white")], transform=ccrs.PlateCarree())
  plt.scatter(x=village.geometry.x, y=village.geometry.y, color='gray', s=12, edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree())

# data_n = data_i.loc[data_i[var]>0,:]

 plt.scatter(x=data_n.geometry.x, y=data_n.geometry.y, c=data_n[var].values, s=13, marker='s', transform=ccrs.PlateCarree(), zorder=0, cmap='magma', vmin=0, vmax=5.5e-07)
 plt.savefig('img/'+label+'/features_'+var+'_'+label+'.png',dpi=300,bbox_inches='tight')
 print('img/'+label+'/features_'+var+'_'+label+'.png')
 plt.close()

