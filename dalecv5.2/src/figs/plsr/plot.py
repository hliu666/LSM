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
    
    pls = PLSRegression(n_components=15)
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

out_x = joblib.load("../../../figs/plsr/out_x.pkl")
out_y = joblib.load("../../../figs/plsr/out_y.pkl")

arr_x = np.array(out_x)
arr_y = np.array(out_y)
#n_comp = 100
#optimise_pls_cv(arr_x, arr_y, n_comp)
                
feature_importance = random_CV(arr_x, arr_y)
wavelength = np.linspace(400.0, 2200.0, num = 1800)
fig, ax = plt.subplots(1, 1, figsize=(22,8))

def_fontsize = 30
def_linewidth = 3.5

linewidth = 1.8 #边框线宽度
ftsize = 20 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度

ax.set_title("GPP/PAR", fontsize=def_fontsize/1.1)
ax.plot(wavelength, feature_importance,  linewidth=def_linewidth, label="DRF = 0.0", color="black")
ax.set_xlabel('Wavelength (nm)'.format(chr(956)), fontsize=def_fontsize)
ax.set_ylabel('Variable Importance in Projection'.format(chr(956)), fontsize=def_fontsize)
#ax[0].legend(loc='lower left', fancybox = False, shadow = False,frameon = False, ncol = 1, fontsize=def_fontsize/1.5)  

ax.spines['left'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
plot_path = "../../../figs/plsr/plsr.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    

