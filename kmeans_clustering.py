import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('bc.txt', sep=",", header=None)
df.head()
df1 = df.drop([0,10],axis=1)
data = np.array(df1)

def euclidean_distance(x,y, along=1):
    #return scipy.spatial.distance.cdist(x,y,'euclidean')
    return np.linalg.norm(x-y,axis=along)

def find_converging_centroid(k):
    clusters = np.zeros(data.shape[0])
    C = np.random.uniform(low=np.min(data), high=np.max(data), size=(k,data.shape[1]))
    C.shape
    C_previous = np.zeros(C.shape)
    change_in_centroid = euclidean_distance(C,C_previous,None)
    while change_in_centroid !=0:
        #assigning cluster
        for i in range(data.shape[0]):
            distance_X_allC = euclidean_distance(data[i],C)
            cluster_index = np.argmin(distance_X_allC)
            clusters[i] = cluster_index
        C_previous = np.array(C, copy=True)
        #finding new centroid points
        for i in range(k):
            x_of_single_cluster = []
            for h in range(data.shape[0]):
                if clusters[h] == i:
                    x_of_single_cluster.append(data[h])
            C[i] = np.mean(x_of_single_cluster,axis=0)
        change_in_centroid = euclidean_distance(C, C_previous, None)
     #   print("itereation ",change_in_centroid)
    return C,clusters

k_list = [2,3,4,5,6,7,8]
t =0
L = [0]* len(k_list)
for k in k_list:
    C,clusters = find_converging_centroid(k)
    for i in range(data.shape[0]):
        L[t] = L[t] + np.sum(np.square(C[int(clusters[i])]-data[i]))
    t = t+1
    print("L = ",L," & K = ",k)

plt.plot(L, label="Testing Error")
plt.xticks(range(len(k_list)),k_list)
plt.xlabel('Different K values')
plt.ylabel('Accumulated Distance from centroid')
plt.title('K-Means k versus distance')
plt.legend()
plt.show()