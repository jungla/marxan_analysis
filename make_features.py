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

# features layers 

label = 'TUMCA'

# fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Pemba/coral_reef/CoralNet_Coordinates_24may22.shp' # Misali
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/HB_CR_Survey/TUMCA/CoralNet_percentages_TUMCA.shp'
data = gpd.read_file(fname)

# deep stations to fill gaps in stations > 50, this should be taken from the original survey design
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/CoralReef/HB_CR_Survey/TUMCA/Deep_coords_TUMCA.shp'
coords = gpd.read_file(fname)
coords = coords.to_crs('EPSG:4326')

data = pd.concat([coords, data])
data = data.drop(['id','Area','Area_ID','UID'],axis=1)

data.loc[data['Row Labels'] != data['Row Labels'],'Sand'] = 1 # fill deep stations with Sand
data = data.fillna(0) # fill everything else with 0

data['Depth'] = abs(data['Depth'].astype('float')) # set positive depth values

data = data.to_crs('epsg:21037')

out_fn = './MarxanData/'+label+'/features_orig.csv'
data.to_csv(out_fn)

features = ['HC','Soft coral','Seagrass','Rubble','Depth']

''' features from coral reef survey '''

fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/grids/TUMCA/marxan_grid_TUMCA_500m.shp'
marxan_grid = gpd.read_file(fname)

ilons = marxan_grid.geometry.x
ilats = marxan_grid.geometry.y

lons = data.geometry.x
lats = data.geometry.y

data_i = data.iloc[:0]
data_i['geometry'] = marxan_grid['geometry']
data_i['labels'] = data_i.index+1

for var in features:
 var_i = griddata((lons,lats), data[var], (ilons, ilats), method='nearest')
 data_i[var] = (var_i-np.min(var_i))/np.max(var_i-np.min(var_i))


''' features from fshery mapping '''

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

''' save data_i to csv '''

data_i = data_i.to_crs('epsg:4326')
out_fn = './MarxanData/'+label+'/features.csv'
data_i.to_csv(out_fn)

''' save input files for Marxan: puvspr [species, pu, amount] '''

species_l = ['HC','Seagrass','Sharks','Rays','Turtles','Mammals']

species = pd.DataFrame() 

for sp in range(len(species_l)):

 species_n = data_i[['labels',species_l[sp]]].rename(columns={species_l[sp]: "amount",'labels':'pu'})
 species_n.insert(2,'species',value=sp+1)

 species = species.append(species_n)

species = species.astype({'pu':int})
species = species[['species','pu','amount']]
species.to_csv('MarxanData/'+label+'/input/puvspr.dat',index=False, sep=',')

