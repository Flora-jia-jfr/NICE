# for learning
import numpy as np

data = np.load("/home/flora_jia/NICE/dat/exp3/replication_0_ep_0.2.npz")
for key in data.keys():
    print("key: ", key)
    print(data[key].shape)

