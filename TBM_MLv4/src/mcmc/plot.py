import spotpy
import pandas as pd
import numpy as np
results = spotpy.analyser.load_csv_results('SCEUA')

import matplotlib.pyplot as plt


bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)

best_model_run = results[bestindex]

fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
best_simulation = list(best_model_run[fields])

df_input = pd.read_csv("../../data/par_set1/HARV.csv")
df_input = df_input[df_input['year'] < 2021]

fig= plt.figure(figsize=(16,9))
ax = plt.subplot(1,1,1)
ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
ax.plot(np.array(df_input['NEE_obs'])[0:16896],'r.',markersize=3, label='Observation data')
plt.xlabel('Number of Observation Points')
plt.ylabel ('Discharge [l s-1]')
plt.legend(loc='upper right')
plt.show()
