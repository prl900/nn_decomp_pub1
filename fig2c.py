from matplotlib import pyplot as plt
import pylab
import numpy as np
import xarray as xr

ds = xr.open_dataset("/home/pl5189/github/sat_decomp/Murrumbidgee_near_Bundure__MUR_B3.nc")
ds = ds.isel(x=slice(400,800), y=slice(0,400))

ds = ds.where((ds.nbart_blue - ds.nbart_blue.quantile(0.25, dim='time'))<1000)
ds = ds.isel(time=(np.count_nonzero(~np.isnan(ds.nbart_blue.values), axis=(1,2)))>400*400*.66)
ds = ds.rolling(time=7, min_periods=3, center=True).median()
ds = ds.reindex({"time": ds.time})
ds = ds.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

stack = np.empty((0,400,400))

for i, band in enumerate(ds):
    stack = np.append(stack, np.clip(ds[band].values / 1e4, 0, 1), axis=0)
    
stack = stack.reshape(-1, 400*400)

plt.imshow(stack, aspect=80, interpolation='none', cmap='bwr')

plt.savefig("fig2c.png", bbox_inches='tight')
