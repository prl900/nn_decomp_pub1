import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import remove_small_objects

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MF(nn.Module):
    def __init__(self, n_coeffs, n_comps, n_pix):
        super(MF, self).__init__()
        self.cfs = nn.Parameter(torch.rand(n_coeffs, n_comps, requires_grad=True))
        self.cmps = nn.Parameter(torch.rand(n_comps, n_pix*n_pix, requires_grad=True))

    def forward(self):
        return torch.matmul(self.cfs, self.cmps)

class MFConv(nn.Module):
    def __init__(self, n_coeffs, n_comps, n_pix):
        super(MFConv, self).__init__()
        self.n_comps = n_comps
        self.n_pix = n_pix
        self.cfs = nn.Parameter(torch.rand(n_coeffs, 2*n_comps, requires_grad=True))
        self.conv1 = nn.Conv2d(n_comps, 2*n_comps, kernel_size=5)
        self.cmps = nn.Parameter(torch.rand(1, n_comps, n_pix+4, n_pix+4, requires_grad=True))

    def forward(self):
        return torch.matmul(self.cfs, self.conv1(self.cmps).view(2*self.n_comps,self.n_pix**2))


def nan_mse_loss(output, target):
    loss = torch.mean((output[target == target] - target[target == target])**2)
    return loss


print("i,j,method,n_comps,band,mse")
for j in range(7,8):
    for i in range(10,25):
        ds = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_clean.nc")
        ds = ds.isel(time=(np.count_nonzero(~np.isnan(ds.nbart_blue.values), axis=(1,2)))>160000*.66)
        
        for ncomps in [6,8,10,12,14,16,18,20]:
            for band_name in ["nbart_red","nbart_green","nbart_blue",
                              "nbart_nir_1","nbart_nir_2", "nbart_swir_2","nbart_swir_3"]:

                stack = ds[band_name].astype(np.float32).values
                stack = stack.reshape(stack.shape[0], -1)
        
                tmean = np.nanmean(stack, axis=0)
                target = torch.from_numpy(stack-tmean).float().to(device)
               
                ncoeffs = stack.shape[0]
                npix = 400

                net = MF(ncoeffs, ncomps, npix)
                net.to(device)
                opt = optim.AdamW(net.parameters(), lr=1.0)
                n_epoch  = 500
                for epoch in range(n_epoch):
                    yhat = net()
                    loss = nan_mse_loss(yhat, target)

                    net.zero_grad()
                    loss.backward()
                    opt.step()

                opt = optim.AdamW(net.parameters(), lr=0.001)
                n_epoch  = 500
                for epoch in range(n_epoch):
                    yhat = net()
                    loss = nan_mse_loss(yhat, target)

                    net.zero_grad()
                    loss.backward()
                    opt.step()

                rec = net().cpu().detach().numpy()+tmean
                print(f"{i},{j},nn_ss,{ncomps},{band_name},{np.nanmean(np.square(rec/10e4-stack/10e4))}", flush=True)

            stack = np.empty((0,400,400))
            for band_name in ["nbart_red","nbart_green","nbart_blue",
                              "nbart_nir_1","nbart_nir_2", "nbart_swir_2","nbart_swir_3"]:

                band = ds[band_name].astype(np.float32).values
                stack = np.append(stack, band, axis=0)

            stack = stack.reshape(stack.shape[0], -1)

            tmean = np.nanmean(stack, axis=0)
            target = torch.from_numpy(stack-tmean).float().to(device)

            ncoeffs = stack.shape[0]
            npix = 400

            net = MF(ncoeffs, ncomps, npix)
            net.to(device)
            opt = optim.AdamW(net.parameters(), lr=1.0)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            opt = optim.AdamW(net.parameters(), lr=0.001)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            rec = net().cpu().detach().numpy()+tmean
            print(f"{i},{j},nn_ms,{ncomps},all,{np.nanmean(np.square(rec/10e4-stack/10e4))}", flush=True)
            
            t_dim = ds.time.shape[0]
            for band_pos, band_name in enumerate(["nbart_red","nbart_green","nbart_blue",
                              "nbart_nir_1","nbart_nir_2", "nbart_swir_2","nbart_swir_3"]):
                print(f"{i},{j},nn_ms,{ncomps},{band_name},{np.nanmean(np.square(rec[band_pos*t_dim:(band_pos+1)*t_dim]/10e4-stack[band_pos*t_dim:(band_pos+1)*t_dim]/10e4))}", flush=True)
            
            net = MFConv(ncoeffs, ncomps, npix)
            net.to(device)
            opt = optim.AdamW(net.parameters(), lr=0.1)
            n_epoch  = 1500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            opt = optim.AdamW(net.parameters(), lr=0.001)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            rec = net().cpu().detach().numpy()+tmean
            print(f"{i},{j},nn_msc,{ncomps},all,{np.nanmean(np.square(rec/10e4-stack/10e4))}", flush=True)
            
            t_dim = ds.time.shape[0]
            for band_pos, band_name in enumerate(["nbart_red","nbart_green","nbart_blue",
                              "nbart_nir_1","nbart_nir_2", "nbart_swir_2","nbart_swir_3"]):
                print(f"{i},{j},nn_msc,{ncomps},{band_name},{np.nanmean(np.square(rec[band_pos*t_dim:(band_pos+1)*t_dim]/10e4-stack[band_pos*t_dim:(band_pos+1)*t_dim]/10e4))}", flush=True)
