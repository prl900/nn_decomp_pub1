import matplotlib.pyplot as plt

import fiona
from shapely import geometry
from shapely.ops import transform
import pyproj

import cartopy.crs as ccrs


project = pyproj.Transformer.from_proj(
    pyproj.Proj('epsg:28355'),
    pyproj.Proj('epsg:3577'))

shape = fiona.open("ACT_Boundary.shp")
shp_geom = transform(project.transform, geometry.shape(next(iter(shape))['geometry']))

aaea = ccrs.AlbersEqualArea(central_latitude=0,
                            false_easting=0,
                            false_northing=0,
                            central_longitude=132,
                            standard_parallels=(-18, -36) )

fig = plt.figure(figsize=(8, 12))
ax = fig.add_subplot(1, 1, 1, projection= aaea)

ax.set_extent([148.7, 149.45, -35.97, -35.07])
ax.coastlines()
ax.gridlines(draw_labels=True)

ax.add_geometries([shp_geom], color='green', alpha=0.8, crs=aaea)

x0 = 1510645.0
y0 = -3936965
nx = 16
ny = 23

for i in range(nx+1):
    geom = geometry.LineString([(x0+i*4000, y0), (x0+i*4000, y0-ny*4000)])
    ax.add_geometries([geom], crs=aaea, color='black')
    
for j in range(ny+1):
    geom = geometry.LineString([(x0, y0-j*4000), (x0+nx*4000, y0-j*4000)])
    ax.add_geometries([geom], crs=aaea, color='black')

#plt.title('DEA Sentinel 2 tiles for the ACT region (Australia)', {'fontsize':30}, pad=40)

plt.savefig("fig1.png")
plt.show()
