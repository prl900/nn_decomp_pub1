from matplotlib import pyplot as plt
import pylab
import numpy as np
import xarray as xr


ds = xr.open_dataset(f"/data/pca_act/000_2018.nc")

rgbs = []
for t in [2,15,38,57]:
    rgb = ds[['nbart_red','nbart_green','nbart_blue']].isel(time=t).to_array().values / 1e4
    rgb = np.moveaxis(rgb, 0, -1)
    rgbs.append(np.clip(rgb*10, 0, 1))

plt.imshow(rgbs[1])

x, y = pylab.ogrid[:400, :400]
ax = pylab.gca(projection='3d')
ax.plot_surface(x, y, np.ones(rgbs[0].shape[:2])*4, rstride=1, cstride=1, facecolors=rgbs[0], shade=False)
ax.plot_surface(x, y, np.ones(rgbs[0].shape[:2])*3, rstride=1, cstride=1, facecolors=rgbs[1], shade=False)
ax.plot_surface(x, y, np.ones(rgbs[0].shape[:2])*2, rstride=1, cstride=1, facecolors=rgbs[1], shade=False)
ax.plot_surface(x, y, np.ones(rgbs[0].shape[:2])*0, rstride=1, cstride=1, facecolors=rgbs[2], shade=False)
ax.set_zticklabels([0,"","...","",63,"",64,"",65])

plt.savefig("fig2a.png")
