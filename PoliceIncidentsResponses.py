%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("d:/WorkFolder/pdata.3/Seattle_Police_Department_911_Incident_Response.csv", low_memory=False)

gr = data.groupby("Event Clearance Description").size()
data["EventDateTime"] = pd.to_datetime(data["Event Clearance Date"])
data["EventYear"] = data['EventDateTime'].apply(lambda x: x.year)
data["EventHour"] = data['EventDateTime'].apply(lambda x: x.hour)
gby = data.groupby("EventYear").size()
gbh = data.groupby("EventHour").size()

plt.subplot(2,1,1)
#plt.tight_layout()
plt.plot(gbh)
plt.xlabel("Hour")
plt.ylabel("Count")
plt.title("Incidents per hour")
plt.subplot(2,1,2)
plt.plot(gby)
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Incidents per year")
plt.tight_layout()
plt.show()

