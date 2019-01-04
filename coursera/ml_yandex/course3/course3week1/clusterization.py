import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

with open('checkins.dat', 'r') as file1:
    with open('checkins-new.dat', 'w') as file2:
        for index, line in enumerate(file1):
            newLine = line.replace(' ','').replace('|', ',')
            _ = file2.write(newLine)  

data = pd.read_csv('checkins-new.dat').dropna()

print(data.head())
print(data.info)

reduced_data = data.iloc[:100000, [3, 4]]

print(reduced_data.head())
print(reduced_data.info)

model = MeanShift(bandwidth=0.1, n_jobs=4)
model.fit(reduced_data)

labels = model.labels_
cluster_centers = model.cluster_centers_

print(labels)
print(cluster_centers)

unique, counts = np.unique(labels, return_counts=True) 
center_candidates = cluster_centers[unique[counts > 15]]

with open('center_candidates', 'w') as centres:
    for center in center_candidates:
        _ = centres.write(str(center[0]) + ',' + str(center[1]) + '\n')

offices = np.array([
    [33.751277, -118.188740],
    [25.867736, -80.324116],
    [51.503016, -0.075479],
    [52.378894, 4.885084],
    [39.366487, 117.036146],
    [-33.868457, 151.205134]])

for office in offices:
    center = 9999999
    centerDistance = 9999999
    for index, center_candidate in enumerate(center_candidates):
        dist = np.linalg.norm(office - center_candidate)
        #print('office: ' + str(office) + ' candidate:' + str(center_candidate) + ' dist:' + str(dist))
        if (dist < centerDistance):
            center = index
            centerDistance = dist
            print('newDist:' + str(centerDistance) + ' center:' + str(center_candidate))

print(center)
print(center_candidates[center])            
#-33.86063043 151.20477593   correct answer