import matplotlib.pyplot as plt
import numpy as np
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import geopandas as gpd
import pandas as pd
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import rasterio
from rasterio.enums import MergeAlg
from functools import partial
from geocube.rasterize import rasterize_image
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial
from geocube.api.core import make_geocube
import rioxarray 
from scipy.interpolate import griddata

label = 'TUMCA'

''' locations at which calculate cost function '''

fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/grids/TUMCA/marxan_grid_TUMCA_500m.shp'
cost_grid = gpd.read_file(fname)


''' effort in TUMCA I calculate from raw FPM results and perceived levels of effort '''

gears = ['Jarife','Juya','Madema','Mishipi','Mtando','Pweza']

wgts = [0.1,0.01,0.32,0.52,0.02,0.01] # proportion of the ppl involved in the fishery 

# attributes 'f_KuN', 'f_KaN', 'Mammals', 'Sea turtles', 'Sharks', 'Rays', 'Permanent', 'Temporary', 'None'

# prepare for interpolation

# extract points at stations
lons, lats = np.meshgrid(cost_grid.geometry.x,cost_grid.geometry.y)

f_gear_t = np.zeros(len(cost_grid.geometry.y))

for g in range(len(gears)):
 f_gear = rioxarray.open_rasterio('/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/FPM/'+gears[g]+'.nc')

 interp = scipy.interpolate.RegularGridInterpolator((f_gear.y.values, f_gear.x.values), np.squeeze(f_gear['f_KuN'].values), bounds_error=False, fill_value=0, method='linear')
 f_Ku = interp((cost_grid.geometry.y, cost_grid.geometry.x))
 interp = scipy.interpolate.RegularGridInterpolator((f_gear.y.values, f_gear.x.values), np.squeeze(f_gear['f_KaN'].values), bounds_error=False, fill_value=0, method='linear')
 f_Ka = interp((cost_grid.geometry.y, cost_grid.geometry.x))

 f_Ku[f_Ku != f_Ku] = 0
 f_Ka[f_Ka != f_Ka] = 0

 f_gear_ku = pd.DataFrame({'effort': f_Ku, 'latitude': cost_grid.geometry.y, 'longitude': cost_grid.geometry.x})
 f_gear_ku.to_csv('./MarxanData/'+label+'/effort_'+gears[g]+'_Ku.csv')
 f_gear_ka = pd.DataFrame({'effort': f_Ka, 'latitude': cost_grid.geometry.y, 'longitude': cost_grid.geometry.x})
 f_gear_ka.to_csv('./MarxanData/'+label+'/effort_'+gears[g]+'_Ka.csv')

 f_gear_g = (f_Ka*0.37 + f_Ku*0.63)*wgts[g] # Ku, 7.5mo, 63% - Ka, 4.5 mo, 37% 
 f_gear_t = f_gear_t + f_gear_g/len(gears) 

 f_gear_o = pd.DataFrame({'effort': f_gear_g, 'latitude': cost_grid.geometry.y, 'longitude': cost_grid.geometry.x})
 f_gear_o.to_csv('./MarxanData/'+label+'/effort_'+gears[g]+'.csv')

f_gear_o = pd.DataFrame({'effort': f_gear_t, 'latitude': cost_grid.geometry.y, 'longitude': cost_grid.geometry.x})
f_gear_o = gpd.GeoDataFrame(f_gear_o, geometry=gpd.points_from_xy(f_gear_o.longitude, f_gear_o.latitude), crs="EPSG:21037")
f_gear_o['labels'] = f_gear_o.index+1

out_fn = './MarxanData/'+label+'/cost_function.csv'
f_gear_o.to_csv(out_fn)

''' create marxan cost layer '''

# normalize cost function from 0 to 1, times a coefficient C

C = 1

f_gear_o['effort'] = (f_gear_o['effort']-np.min(f_gear_o['effort']))/np.max(f_gear_o['effort']-np.min(f_gear_o['effort']))*C

#output = pd.DataFrame({'id':f_gear_o.labels.values,'cost':f_gear_o.effort.values,'status':np.zeros(len(f_gear_o))})
output = pd.DataFrame({'id':f_gear_o.labels.values,'cost':f_gear_o.effort.values,'status':np.zeros(len(f_gear_o)), 'xloc': f_gear_o.longitude,'yloc': f_gear_o.latitude})
output.to_csv('MarxanData/'+label+'/input/pu.dat',index=False, sep=',')

''' create bounding file '''

f_gear_o = f_gear_o.to_crs('EPSG:21037')
f_gear_o = f_gear_o.astype({'labels':'int'})
dist = f_gear_o.geometry.apply(lambda g: f_gear_o.distance(g))
dist = dist[dist < 700]
dist = dist[dist > 0]

id1 = []
id2 = []

for i in range(len(f_gear_o)):
# data[~np.isnan(dist[i])]
 val = f_gear_o[~np.isnan(dist[i])].labels.values
 id1.extend(list(np.zeros(len(val))+f_gear_o['labels'][i]))
 id2.extend(list(val))

output = pd.DataFrame({'id1':id1,'id2':id2,'boundary':np.zeros(len(id1))+1})
output = output.astype({'id1':'int'})

cols = ['id1', 'id2']
output[cols] = np.sort(output[cols].values, axis=1)
output = output.drop_duplicates()

output.to_csv('MarxanData/'+label+'/input/bound.dat',index=False, sep=',')
