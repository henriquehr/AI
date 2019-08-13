import os
import matplotlib.pyplot as plt

metrics = { 
    'ite':[], 
    'loss_output':[], 
    'loss_pre_ref':[],
}

print('loading logs...')
with open('logs/log_val.txt') as file:
    for l in file:
        if '[val:' in l[:8]:
            w = l.replace(':', '').replace(']','').replace(',','').split(' ')
            for i, x in enumerate(w):
                if x in metrics:
                    if x == 'ite':
                        if float(w[i+1]) in metrics['ite']:
                            break
                    metrics[x].append(float(w[i+1]))

print('plotting...')
for k in metrics.keys():
    if k == 'ite':
        continue
    print(metrics[k])
    plt.plot(metrics['ite'], metrics[k])
    plt.title(k)
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.show()
