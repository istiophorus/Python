import pygal
import urllib3
import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

fname = "d:/googticks.txt"

def download_and_save(file_name):
    url = "https://www.google.com/finance/getprices?q=GOOG&i=8&p=1d&f=d,o,h,l,c,v&df=cpct"
    urllib3.disable_warnings()
    http = urllib3.PoolManager()
    r = http.request('GET', url)

    with open(file_name, "wb") as f:
        f.write(r.data)    


def read_data(file_name):
    content = ""
    with open(file_name, 'r') as content_file:
        content = content_file.read()
        
    lines = content.split("\n")
    
    result = []
    
    for line in lines:
        if line.startswith("a"):
            splitted = line.split(",")
            if len(splitted) == 6:
                ts = int(splitted[0].lstrip("a"))
                dtime = dt.datetime.utcfromtimestamp(ts)
                dx = (ts, dtime, float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4]), float(splitted[5]))
                
                result.append(dx)
                
    data = pd.DataFrame(result, columns=["Timestamp", "UTC", "CLOSE","HIGH","LOW","OPEN","VOLUME"])
                
    return data


if not os.path.isfile(fname):
    download_and_save(fname)

data = read_data(fname)            

plt.figure(figsize=(6,3), dpi=200)
plt.plot(data.UTC, data.CLOSE)
plt.plot(data.UTC, data.HIGH)
plt.plot(data.UTC, data.LOW)
plt.show()