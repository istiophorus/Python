%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cs = pd.read_csv("d:/WorkFolder/pdata.4/countries.csv", low_memory=False)

cs["GDP"] = cs.gdpPerCapita * cs.population

d2007 = cs[cs.year == cs.year.max()]
d2007s = d2007.sort_values("GDP", ascending=False)
gdp10 = d2007s.head(10)
ixs = list(range(len(gdp10)))

plt.figure(figsize=(4,6), dpi=200)

plt.subplot(3,1,1)
plt.bar(ixs,gdp10["GDP"])
plt.xticks(ixs, gdp10.continent, rotation="vertical")
plt.title("Top 10 GDP")

gdppc10 = d2007.sort_values("gdpPerCapita", ascending=False).head(10)
plt.subplot(3,1,2)
plt.bar(ixs,gdppc10["gdpPerCapita"])
plt.title("Top 10 GDP per capita")
plt.xticks(ixs, gdppc10.continent, rotation="vertical")

pop10 = d2007.sort_values("population", ascending=False).head(10)
plt.subplot(3,1,3)
plt.title("Top 10 population")
plt.bar(ixs,pop10["population"])
plt.xticks(ixs, pop10.continent, rotation="vertical")

plt.tight_layout()
plt.show()