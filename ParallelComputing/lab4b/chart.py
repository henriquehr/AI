import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
 

    cuda = "times.txt"
    cuda_shm = "timesSHM.txt"
    

    time_cuda = []
    time_cuda_shm = []
    with open(cuda) as file:
        txt = file.read().split("\n")
        time_cuda = [float(f) for f in txt if f is not ""]

    with open(cuda_shm) as file:
        txt = file.read().split("\n")
        time_cuda_shm = [float(f) for f in txt if f is not ""]

    speedup = []
    for i in range(len(time_cuda)):
        speedup.append(time_cuda[i] / time_cuda_shm[i])
 
    fig, ax = plt.subplots()

    bar_width = 0.2

    ax.bar(np.arange(len(time_cuda)), time_cuda, bar_width, label="Blur")

    ax.bar(np.arange(len(time_cuda_shm)) + bar_width, time_cuda_shm, bar_width, label="BlurSHM")

    ax2 = ax.twinx()
    ax2.set_ylabel("Speedup", color="r")
    ax2.tick_params("y", colors="r")
    ax2.plot(speedup, color="r")

    ax.set_xlabel('File')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(np.arange(len(time_cuda)) + bar_width / 2)
    ax.set_xticklabels(["W0","W1",0,1,3,4,5,6,7,8,9])
    ax.legend()

    fig.tight_layout()
    plt.show()
    