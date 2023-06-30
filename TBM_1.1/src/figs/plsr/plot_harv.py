# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 20:54:56 2022

@author: hliu
"""
from sys import stdout

import joblib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

def optimise_pls_cv(X, y, n_comp, plot_components=True):
 
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''
 
    mse = []
    component = np.arange(1, n_comp)
 
    for i in component:
        pls = PLSRegression(n_components=i)
 
        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=10)
 
        mse.append(mean_squared_error(y, y_cv))
 
        comp = 100*(i+1)/n_comp
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
 
    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)
    stdout.write("\n")
 
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=-1)
 
        plt.show()
 
    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin+1)
 
    # Fir to the entire dataset
    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)
 
    # Cross-validation
    y_cv = cross_val_predict(pls_opt, X, y, cv=10)
 
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
 
    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
 
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
 
    # Plot regression and figures of merit
    rangey = max(y) - min(y)
    rangex = max(y_c) - min(y_c)
 
    # Fit a line to the CV vs response
    z = np.polyfit(y, y_c, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_c, y, c='red', edgecolors='k')
        #Plot the best fit line
        ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
        #Plot the ideal 1:1 line
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): '+str(score_cv))
        plt.xlabel('Predicted $^{\circ}$Brix')
        plt.ylabel('Measured $^{\circ}$Brix')
 
        plt.show()
 
    return

def random_CV(X, y):
    
    pls = PLSRegression(n_components=30)
    pls.fit(X, y)
    
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T,t),q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(np.matmul(s.T, weight))/total_s)
        
    return vips[0:1800]

out_x = joblib.load("../../../data/output/model/brf_ci1_HARV.pkl")

arr_x = np.array(out_x)

out_gpp = joblib.load("../../../data/output/model/An_ci1_HARV.pkl")
out_apar = joblib.load("../../../data/output/model/apar_ci1_HARV.pkl")
out_par = joblib.load("../../../data/output/model/par_ci1_HARV.pkl")

arr_gpp = np.array(out_gpp)
arr_apar = np.array(out_apar)
arr_par = np.array(out_par)

idx = np.where(arr_gpp < 0.5)[0]

arr_x = np.delete(arr_x, idx, axis=0)

arr_gpp = np.delete(arr_gpp, idx, axis=0)
arr_apar = np.delete(arr_apar, idx, axis=0)
arr_par = np.delete(arr_par, idx, axis=0)

arr_y1 = (arr_gpp/arr_apar).flatten()
arr_y2 = (arr_gpp/arr_par).flatten()

data = pd.read_csv("../../../data/parameters/HARV_spectral.csv", na_values="nan")
refl = np.array(data['refl'])[0:1800]
#n_comp = 99
#optimise_pls_cv(arr_x, arr_y, n_comp)
                
feature_importance1 = random_CV(arr_x, arr_y1)
feature_importance2 = random_CV(arr_x, arr_y2)
feature_importance3 = random_CV(arr_x, arr_gpp)

feature_importance = np.hstack((feature_importance1.reshape(-1,1), feature_importance2.reshape(-1,1), feature_importance3.reshape(-1,1)))
df = pd.DataFrame(feature_importance) 
df.to_csv("../../../figs/plsr/feature_importance.csv",header=False,index=False)

wavelength = np.arange(400.0, 2200.0, 1.0)
fig, ax = plt.subplots(3, 1, figsize=(22,24))

def_fontsize = 35
def_linewidth = 3.5

linewidth = 2.0 #边框线宽度
ftsize = 35 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度

ax[0].set_title("(a) GPP/PAR", fontsize=def_fontsize*1.2)
ax[0].plot(wavelength, feature_importance1,  linewidth=def_linewidth, color="black")
ax[0].axhline(y = 1.0, color='red', lw=def_linewidth, ls='--')
ax[0].set_xlim(400, 2200)
ax[0].set_ylim(0, 2)

ax0 = ax[0].twinx()
ax0.plot(wavelength, refl, color='green', lw=def_linewidth, ls='--')

ax[1].set_title("(b) GPP/APAR", fontsize=def_fontsize*1.2)
ax[1].plot(wavelength, feature_importance2,  linewidth=def_linewidth, color="black")
ax[1].axhline(y = 1.0, color='red', lw=def_linewidth, ls='--')
ax[1].set_xlim(400, 2200)
ax[1].set_ylim(0, 2)

ax1 = ax[1].twinx()
ax1.plot(wavelength, refl, color='green', lw=def_linewidth, ls='--')

ax[2].set_title("(c) GPP", fontsize=def_fontsize*1.2)
ax[2].plot(wavelength, feature_importance3,  linewidth=def_linewidth, color="black")
ax[2].axhline(y = 1.0, color='red', lw=def_linewidth, ls='--')
ax[2].set_xlim(400, 2200)
ax[2].set_ylim(0, 2)

ax2 = ax[2].twinx()
ax2.plot(wavelength, refl, color='green', lw=def_linewidth, ls='--')

ax0.yaxis.label.set_color('green')
ax1.yaxis.label.set_color('green')
ax2.yaxis.label.set_color('green')

ax[0].spines['left'].set_linewidth(linewidth)
ax[0].spines['right'].set_linewidth(linewidth)
ax[0].spines['top'].set_linewidth(linewidth)
ax[0].spines['bottom'].set_linewidth(linewidth)
ax[0].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)
        
ax[1].spines['left'].set_linewidth(linewidth)
ax[1].spines['right'].set_linewidth(linewidth)
ax[1].spines['top'].set_linewidth(linewidth)
ax[1].spines['bottom'].set_linewidth(linewidth)
ax[1].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
 
ax[2].spines['left'].set_linewidth(linewidth)
ax[2].spines['right'].set_linewidth(linewidth)
ax[2].spines['top'].set_linewidth(linewidth)
ax[2].spines['bottom'].set_linewidth(linewidth)
ax[2].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
   
ax0.spines['left'].set_linewidth(linewidth)
ax0.spines['right'].set_linewidth(linewidth)
ax0.spines['top'].set_linewidth(linewidth)
ax0.spines['bottom'].set_linewidth(linewidth)
ax0.tick_params(direction = 'in', axis='both', colors= 'green', length = axlength, width = axwidth, labelsize = ftsize)
        
ax1.spines['left'].set_linewidth(linewidth)
ax1.spines['right'].set_linewidth(linewidth)
ax1.spines['top'].set_linewidth(linewidth)
ax1.spines['bottom'].set_linewidth(linewidth)
ax1.tick_params(direction = 'in', axis='both', colors= 'green', length = axlength, width = axwidth, labelsize = ftsize)  
      
ax2.spines['left'].set_linewidth(linewidth)
ax2.spines['right'].set_linewidth(linewidth)
ax2.spines['top'].set_linewidth(linewidth)
ax2.spines['bottom'].set_linewidth(linewidth)
ax2.tick_params(direction = 'in', axis='both', colors= 'green', length = axlength, width = axwidth, labelsize = ftsize)        
   
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
#plt.xlabel("NEON observations", fontsize = ftsize*1.5, family = ftfamily, labelpad=15)
#plt.ylabel("Model simulations", fontsize = ftsize*1.5, family = ftfamily, labelpad=15)

ax[0].set_ylabel('Variable Importance in Projection', fontsize=def_fontsize)
ax[1].set_ylabel('Variable Importance in Projection', fontsize=def_fontsize)
ax[2].set_ylabel('Variable Importance in Projection', fontsize=def_fontsize)

ax0.set_ylabel('Reflectance', fontsize=def_fontsize)
ax1.set_ylabel('Reflectance', fontsize=def_fontsize)
ax2.set_ylabel('Reflectance', fontsize=def_fontsize)

plt.xlabel('Wavelength (nm)', fontsize=def_fontsize, labelpad=20)

fig.tight_layout()          

plot_path = "../../../figs/plsr/plsr.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    

