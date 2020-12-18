from matplotlib import pyplot as plt
import pylab
import numpy as np
import xarray as xr

t = 2
ds = xr.open_dataset(f"/data/pca_act/000_2018.nc")
ds = ds.isel(time=t)

bands = []
scales = [10,10,10,1.5,1.5,2,3]
for i, band in enumerate(ds):
    bands.append(scales[i]*np.clip(ds[band].values / 1e4, 0, 1))

x, y = pylab.ogrid[:400, :400]
ax = pylab.gca(projection='3d')
ax.plot_surface(x, y, np.ones(bands[0].shape[:2])*0, rstride=1, cstride=1, facecolors=plt.cm.Blues_r(bands[0]), shade=False)
ax.plot_surface(x, y, np.ones(bands[0].shape[:2])*1, rstride=1, cstride=1, facecolors=plt.cm.Greens_r(bands[1]), shade=False)
ax.plot_surface(x, y, np.ones(bands[0].shape[:2])*2, rstride=1, cstride=1, facecolors=plt.cm.Reds_r(bands[2]), shade=False)
ax.plot_surface(x, y, np.ones(bands[0].shape[:2])*3, rstride=1, cstride=1, facecolors=plt.cm.gist_heat(bands[3]), shade=False)
ax.plot_surface(x, y, np.ones(bands[0].shape[:2])*4, rstride=1, cstride=1, facecolors=plt.cm.gist_heat(bands[4]), shade=False)
ax.plot_surface(x, y, np.ones(bands[0].shape[:2])*5, rstride=1, cstride=1, facecolors=plt.cm.hot(bands[5]), shade=False)
ax.plot_surface(x, y, np.ones(bands[0].shape[:2])*6, rstride=1, cstride=1, facecolors=plt.cm.hot(bands[6]), shade=False)
ax.set_zticklabels([0,"","","...","","",461])

plt.savefig("fig2b.png", bbox_inches='tight')
