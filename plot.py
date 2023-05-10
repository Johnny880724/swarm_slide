import numpy as np
import csv

result_dict = {"x":0,"swarm": 1, "slide": 2, "mix": 3, "swarm/consolidation": 4}

with open('/Users/btsao/Desktop/UT_Austin/Fall_2022/PHY 380N/transition_data.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     heading = True
     gel_mass_arr, evap_mass_arr, loss_mass_arr = [], [], []
     result_arr, date_arr = [], []
     result_num_arr = []
     row_num = 0
     for row in spamreader:
        if(heading):
            row_headings = row
            heading = False
        elif(row_num in [49,59,97]): #outliers
        # elif(row_num in []):
            print("Do not include row ",row_num)
        else:
            elem = row[0].split(",")
            if(result_dict[elem[3]] != 0):
                gel_mass_arr.append(float(elem[0]))
                evap_mass_arr.append(float(elem[1]))
                loss_mass_arr.append(float(elem[2]))
                result_arr.append(elem[3])
                result_num_arr.append(result_dict[elem[3]])
                date_arr.append(elem[4])
                print(row_num,gel_mass_arr[-1],loss_mass_arr[-1],result_arr[-1]) 
            else:
                print("Discard x data ",row_num)
        row_num += 1

print("there are rows",len(result_num_arr))
import matplotlib.pyplot as plt
gel_mass_dict, loss_mass_dict = {}, {}
keys = ["swarm", "slide", "mix", "swarm/consolidation"]
# keys = ["swarm", "slide"]
for key in keys:
    gel_mass_dict[key] = []
    loss_mass_dict[key] = []

for i in range(len(result_arr)):
    for key in keys:
        if(result_arr[i] == key):
            gel_mass_dict[key].append(gel_mass_arr[i]) 
            loss_mass_dict[key].append(loss_mass_arr[i])
colors = {"swarm": "r", "slide": "b", "mix": "y", "swarm/consolidation": "g"}

fig = plt.figure(figsize = (6,4))
ax = fig.gca()

for key in keys:
    ax.plot(gel_mass_dict[key], loss_mass_dict[key], ".",color = colors[key], label = key)
plt.legend()


param_center = np.array([gel_mass_dict["swarm"]  + gel_mass_dict["slide"], \
                  loss_mass_dict["swarm"] + loss_mass_dict["slide"]]).transpose()
result_num_arr_used = result_dict["swarm"] + result_dict["slide"]
center_target = [False]* len(gel_mass_dict["swarm"]) +[True]* len(gel_mass_dict["slide"])

param = np.array([gel_mass_arr, loss_mass_arr]).transpose()
swarm_target = [result_num in [1] for result_num in result_num_arr]
slide_target = [result_num in [2] for result_num in result_num_arr]

def svc_linear(param, target):
    from sklearn import svm
    xy_arr = np.array(param.transpose())
    x_mean, x_std = np.average(xy_arr[0]),np.std(xy_arr[0])
    y_mean, y_std = np.average(xy_arr[1]),np.std(xy_arr[1])
    
    x_norm = (xy_arr[0] - x_mean)/x_std
    y_norm = (xy_arr[1] - y_mean)/y_std
    
    
    clf = svm.SVC(kernel='linear')
    clf.fit(np.transpose(np.array([x_norm,y_norm])), target)
    # x2 = -w1/w2 x1 - b/w2
    a_norm = - clf.coef_[0,0]/clf.coef_[0,1]
    b_norm = - clf.intercept_[0]/clf.coef_[0,1]
    
    # plot svm
    # plt.figure(figsize=(5,5))
    # for i in range(len(x_norm)):
    #     if(target[i]):
    #         plt.plot(x_norm[i],y_norm[i],"r.")
    #     else:
    #         plt.plot(x_norm[i],y_norm[i],"b.")
    # plt.xlim(-2,2)
    # plt.ylim(-2,2)
    # x = np.linspace(np.min(x_norm),np.max(x_norm),10)
    # plt.plot(x,a_norm*x +b_norm)
    
    a = a_norm * y_std / x_std
    b = (b_norm - a_norm*x_mean/x_std) * y_std + y_mean
    return a, b

buffu = 1.02
buffd = 2 - buffu

x = np.linspace(buffd* np.min(gel_mass_arr), buffu* np.max(gel_mass_arr), 10)

swarm_a, swarm_b = svc_linear(param,swarm_target)
y_swarm          = swarm_a * x + swarm_b
ax.plot(x,y_swarm,color="red")

slide_a, slide_b = svc_linear(param,slide_target)
y_slide          = slide_a * x + slide_b
ax.plot(x,y_slide,color="blue")

plt.fill_between(x,y_swarm,color="red",alpha = 0.3)
plt.fill_between(x,y_slide,y_swarm,color="green",alpha = 0.3)
plt.fill_between(x,y_slide,buffu* np.max(loss_mass_arr),color="blue",alpha = 0.3)

center_a, center_b = svc_linear(param_center,center_target)
y_center          = center_a * x + center_b
ax.plot(x,y_center,color="black")

ax.set_xlabel("gel mass (g)")
ax.set_ylabel("mass loss (%)")


ax.set_xlim(buffd* np.min(gel_mass_arr),buffu* np.max(gel_mass_arr))
ax.set_ylim(0,buffu* np.max(loss_mass_arr))
plt.grid()

fig.savefig('/Users/btsao/Desktop/UT_Austin/Fall_2022/PHY 380N/swarm_slide.pdf')
