# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 23:12:28 2023

@author: hliu
"""
import numpy as np
import joblib
brf = joblib.load("../../../data/output/model/brf_ci1_HARV.pkl") 
#extract reflectance in Aug 26 2019 (doy of 248)   
brf_sim = np.average(brf[(247)*24+10:(247)*24+14,:2100], axis=0)
brf_sim_wvl = np.arange(400, 2500, 1)

from spectral import envi
root = "G:/NEON/verify/7.0.hyperspectral/"
img = envi.open(root+'NEON_D01_HARV_DP1_20190826_142943_reflectance_topo_brdf.hdr')
brf_obs_wvl = img.bands.centers
brf_obs_wvl = np.array(brf_obs_wvl[4:-4]) #select wavelength within 400-2500nm

from spectres import spectres
brf_sim_resample = spectres(brf_obs_wvl, brf_sim_wvl, brf_sim)

from osgeo import gdal_array
src = "clip.tif"
arr = gdal_array.LoadFile(root+src)
arr = arr[4:-4,:,:]
arr = arr.reshape(418, -1).T
arr[arr < 0] = np.nan
arr = arr[~np.any(np.isnan(arr), axis=1),:]
arr = arr[np.where(arr[:, 84] > 0.06)]
arr = arr[np.where(((arr[:,84]-arr[:,57])/(arr[:,84]+arr[:,57])) > 0.6)]
arr = arr/10000.0 #scaling factor is 10,000

brf_obs_mean = np.average(arr, axis=0)
brf_obs_std = np.std(arr, axis=0)

arr_plot = np.hstack((brf_obs_wvl.reshape(-1,1),\
                     brf_sim_resample.reshape(-1,1),\
                     brf_obs_mean.reshape(-1,1),\
                     brf_obs_std.reshape(-1,1)))
   
# 418.59–1335.04 nm 
# 1460.23–1770.72 nm 
# 1986.06–2396.71 nm   
 
#arr_plot = arr_plot[((arr_plot[:,0]>=418.59) & (arr_plot[:,0]<=1335.04)) |
#                    ((arr_plot[:,0]>=1460.23) & (arr_plot[:,0]<=1770.72)) |
#                    ((arr_plot[:,0]>=1986.06) & (arr_plot[:,0]<=2396.71))]

arr_plot[arr_plot[:,0]<418.59, 1:4] = np.nan
arr_plot[((arr_plot[:,0]>1335.04) & (arr_plot[:,0]<1460.23)), 1:4] = np.nan
arr_plot[((arr_plot[:,0]>1986.06) & (arr_plot[:,0]<2396.71)), 1:4] = np.nan

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.transforms import Affine2D
import scipy.stats as stats

fig, axs = plt.subplots(1, 1, figsize=(12, 8))

subplt = axs

linewidth = 2.5  # 边框线宽度
ftsize = 30  # 字体大小
axlength = 8.0  # 轴刻度长度
axwidth = 1.8  # 轴刻度宽度
ftfamily = 'Calibri'

capsize = 2  # 柱状体标准差参数1
capthick = 1  # 柱状体标准差参数2
markersize = 4

arr_plot_nonan = arr_plot[~np.isnan(arr_plot).any(axis=1)]
subplt.annotate('R\u00b2={0},RMSE={1}'.format(round(stats.pearsonr(arr_plot_nonan[:,1], arr_plot_nonan[:,2])[0] ** 2,2),
                                                          round(np.sqrt(((arr_plot_nonan[:,1] - arr_plot_nonan[:,2]) ** 2).mean()), 2)),
                xy=(0.55, 0.7), xycoords='axes fraction', fontsize=ftsize)

subplt.spines['left'].set_linewidth(linewidth)
subplt.spines['right'].set_linewidth(linewidth)
subplt.spines['top'].set_linewidth(linewidth)
subplt.spines['bottom'].set_linewidth(linewidth)
subplt.tick_params(direction='in', axis='both', length=axlength, width=axwidth, labelsize=ftsize)

#subplt.set_title('({0}) {1}, {2}'.format(chr(97 + 0), site_name, site_Lc), fontsize=ftsize,
#                 family=ftfamily)

trans1 = Affine2D().translate(0, 0.0) + subplt.transData
#GPP_PRE = subplt.errorbar(hour_mean.index, hour_mean['GPP_prediction'], yerr=hour_std['GPP_prediction'],
#                          marker="o", markersize=markersize, c='red',
#                          linestyle="-", transform=trans1, capsize=capsize, capthick=capthick)
SIM, = subplt.plot(arr_plot[:,0], arr_plot[:,1], color='blue', lw=linewidth * 1.5)

trans2 = Affine2D().translate(+0.25, 0.0) + subplt.transData
#OBS = subplt.errorbar(arr_plot[:,0], arr_plot[:,2], yerr=arr_plot[:,3],
#                          marker="o", markersize=markersize, c='red',
#                          linestyle="-", transform=trans2, capsize=capsize, capthick=capthick)

subplt.fill_between(arr_plot[:,0], arr_plot[:,2]-arr_plot[:,3], arr_plot[:,2]+arr_plot[:,3], alpha=0.3, color='red')
OBS, = subplt.plot(arr_plot[:,0], arr_plot[:,2], color='red', lw=linewidth * 1.5)

handles = [SIM, OBS]
labels = ['Simulations', 'Observations']

plt.xlabel("Wavelength(nm)", fontsize=ftsize)
plt.ylabel("Canopy Reflectance".format(chr(956)), fontsize=ftsize)

fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           fancybox=False, shadow=False, frameon=False, ncol=2,
           handletextpad=0.2, columnspacing=1.2, prop={'family': ftfamily, 'size': ftsize})

plot_path = "../../../figs/refl/valid_refl.jpg"
plt.show()
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

                