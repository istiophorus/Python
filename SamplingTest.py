%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_big = pd.read_csv("d:/WorkFolder/pdata.4/obama_too_big.csv", low_memory=False, parse_dates=['year_month'])
data = pd.read_csv("d:/WorkFolder/pdata.4/obama.csv", low_memory=False, parse_dates=['year_month'])

sampled = data_big.sample(frac=0.1)

plt.figure(figsize=(6,7), dpi=200)
plt.subplot(2,1,1)
plt.plot(data_big.year_month, data_big.approve_percent, 'o', markersize=2, alpha=.3)
plt.subplot(2,1,2)
plt.plot(sampled.year_month, sampled.approve_percent, 'o', markersize=2, alpha=.3)

plt.show()