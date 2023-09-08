import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import geopandas as gpd
import pandas as pd
import matplotlib.patches as mpatches
import rasterio
from rasterio import features
import numpy as np
 
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


# effort per fishing ground

# effort
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/catch_data/Pemba/effort_ground.csv'
effort_ground = pd.read_csv(fname)

# fishing ground
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Pemba/fishery_mapping/fishing gear shapefiles/all_gears_clean.shp'
fishing_ground = gpd.read_file(fname)

fishing_effort = pd.merge(fishing_ground,effort_ground,left_on='Name', right_on='fishing ground')

# I think I need this before rasterizing
fishing_effort['tripsqkm']=fishing_effort['trips']/fishing_effort['area']*1e6

# rasterize layers before merging them

import osr
import gdal

driver = gdal.GetDriverByName('GTiff')

spatref = osr.SpatialReference()
spatref.ImportFromEPSG(4326)
wkt = spatref.ExportToWkt()

outfn = 'template.tif'
nbands = len(fishing_effort)
nodata = -9999
xres = 0.005
yres = -0.005

xmin = 39.4
ymin = -5.5
xmax = 39.7
ymax = -5.0

dtype = gdal.GDT_Float32

xsize = abs(int((xmax - xmin) / xres))
ysize = abs(int((ymax - ymin) / yres))

ds = driver.Create(outfn, xsize, ysize, nbands, dtype, options=['COMPRESS=LZW', 'TILED=YES'])
ds.SetProjection(wkt)
ds.SetGeoTransform([xmin, xres, 0, ymax, 0, yres])
ds.GetRasterBand(1).Fill(0)
ds.GetRasterBand(1).SetNoDataValue(nodata)
ds.FlushCache()
ds = None

rst_fn = 'template.tif'
out_fn = 'fishing_grounds.tif'

rst = rasterio.open(rst_fn)

meta = rst.meta.copy()
meta.update(compress='lzw')

with rasterio.open(out_fn, 'w+', **meta) as out:
 for b in range(1,nbands+1):
  print(b)
  out_arr = out.read(b)
 
  # this is where we create a generator of geom, value pairs to use in rasterizing
  #shapes = ((geom,value) for geom, value in zip(fishing_effort['geometry'][b], fishing_effort['tripsqkm'][b]))
 
  burned = features.rasterize(shapes=[(fishing_effort['geometry'][b-1], fishing_effort['tripsqkm'][b-1])], fill=-9999, out=out_arr, transform=out.transform)
  out.write_band(b, burned)
out.close()

f = rasterio.open(out_fn)
data = f.read()
data[data == -9999] = np.nan
effort = np.nansum(data,axis=0)

# coords

height = data.shape[1]
width = data.shape[2]
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(f.transform, rows, cols)
lons= np.array(xs)
lats = np.array(ys)

# prepare for interpolation

lons = lons.ravel()
lats = lats.ravel()
effort = effort.ravel()

lons = lons[~np.isnan(effort)]
lats = lats[~np.isnan(effort)]
effort = effort[~np.isnan(effort)]

# extract points at stations
fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Data/Pemba/coral_reef/CoralNet_Coordinates_24may22.shp'
data = gpd.read_file(fname)

# drop empty rows
#data.replace('', np.nan, inplace=True)
#data.dropna(subset = ['geometry'], inplace=True)

# extract points [no need to go back to raster]
from scipy.interpolate import griddata
extr_effort = griddata((lons,lats), effort, (data.Longitude.ravel(), data.Latitude.ravel()), method='nearest')

# plotting

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([39.56,39.77,-5.38,-5.12], crs=ccrs.PlateCarree())

ax.add_feature(shape_bathymetry)
ax.add_feature(shape_coastline)
ax.add_feature(shape_mangroves)
ax.add_feature(shape_coralreefs)

#plt.scatter(x=lons, y=lats, color='red', s=effort*10, alpha=1, zorder=100, transform=ccrs.PlateCarree())
plt.scatter(x=data.Longitude,y=data.Latitude, color='green', s=extr_effort*10, marker='D', edgecolor='black', linewidths=0.5, alpha=1, zorder=100, transform=ccrs.PlateCarree(), label='Fishing effort')
plt.legend()
plt.savefig('img/cost_fishing_effort.png',dpi=300,bbox_inches='tight')
plt.close()

# create cost file for Marxan
output = pd.DataFrame({'id':data.Labels.astype('int'),'cost':extr_effort,'status':np.zeros(len(extr_effort))})
output.to_csv('MarxanData/input/pu.dat',index=False, sep=',')

# craete bound matrix
data = data.to_crs('21037')
data = data.astype({'Labels':'int'})
dist = data.geometry.apply(lambda g: data.distance(g))
dist = dist[dist < 1500]
dist = dist[dist > 0]

id1 = []
id2 = []

for i in range(len(data)):
# data[~np.isnan(dist[i])]
 val = data[~np.isnan(dist[i])].Labels.values
 id1.extend(list(np.zeros(len(val))+data['Labels'][i]))
 id2.extend(list(val))

# create cost file for Marxan
output = pd.DataFrame({'id1':id1,'id2':id2,'boundary':np.zeros(len(id1))+1})
output = output.astype({'id1':'int'})

cols = ['id1', 'id2']
output[cols] = np.sort(output[cols].values, axis=1)
output = output.drop_duplicates()

output.to_csv('MarxanData/input/bound.dat',index=False, sep=',')

