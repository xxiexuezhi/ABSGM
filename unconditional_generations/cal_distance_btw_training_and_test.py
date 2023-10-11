

import pickle 

with open("dataset_antibody_light_heavy_all_info_0_3500_date0831.pkl","rb") as f:
    d1= pickle.load(f)
#with open("dataset_antibody_light_heavy_all_info_3500_7000_date0831.pkl","rb") as f:
#    d2= pickle.load(f)
#with open("dataset_antibody_light_heavy_all_info_7000_10768_date0831.pkl","rb") as f:
#    d3= pickle.load(f)


x =d1# + d2 +d3


print(x[0])
length = int(len(x))

cut_p = int(length * 0.9)


train_data = x[:cut_p]
test_data = x[cut_p:]


