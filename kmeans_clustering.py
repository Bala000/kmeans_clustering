import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

df = pd.read_csv('bc.txt', sep=",", header=None)
df.head()
df1 = df.drop([0,10],axis=1)
data = np.array(df1)


def find_converging_centroid(k):
    clusters = np.zeros(data.shape[0])
    C = np.random.uniform(low=np.min(data), high=np.max(data), size=(k,data.shape[1]))
    print(C)
    C_previous = np.zeros(C.shape)
    change_in_centroid = euclidean_distance(C,C_previous,None)

    while change_in_centroid !=0:
        empty_cluster = False
        empty_cluster_index = 0
        biggest_cluster = 0
        single_biggest_cluster = []
        biggest_cluster_index = 0
        #CLASSIFY
        for i in range(data.shape[0]):
            distance_X_allC = euclidean_distance(data[i],C)
            cluster_index = np.argmin(distance_X_allC)
            clusters[i] = cluster_index
        C_previous = np.array(C, copy=True)
        #RECENTER
        for i in range(k):
            x_of_single_cluster = []
            for h in range(data.shape[0]):
                if clusters[h] == i:
                    x_of_single_cluster.append(data[h])
            if len(x_of_single_cluster)==0:
                empty_cluster = True
                empty_cluster_index = i
                print("empty cluster found")
            else:
                C[i] = np.mean(x_of_single_cluster,axis=0)
            if biggest_cluster < len(x_of_single_cluster):
                biggest_cluster = len(x_of_single_cluster)
                single_biggest_cluster = copy.copy(x_of_single_cluster)
                biggest_cluster_index = i
        if empty_cluster:
            #divide the biggest cluster into two equal cluster
            list1 = single_biggest_cluster[:int(len(single_biggest_cluster)/2)]
            list2 = single_biggest_cluster[int(len(single_biggest_cluster)/2):]
            C[biggest_cluster_index] = np.mean(list1,axis=0)
            C[empty_cluster_index] = np.mean(list2,axis=0)
        change_in_centroid = euclidean_distance(C, C_previous, None)
    return C,clusters

def euclidean_distance(x,y, along=1):
    #return scipy.spatial.distance.cdist(x,y,'euclidean')
    return np.linalg.norm(x-y,axis=along)


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
plt.ylabel('(Potential Function) Accumulated Distance from centroid')
plt.title('K-Means k versus distance')
plt.legend()
plt.show()