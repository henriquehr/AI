import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
 
    K = sys.argv[1]
    iters = sys.argv[2]
    K2 = sys.argv[3]
    iters2 = sys.argv[4]
    K3 = sys.argv[5]
    iters3 = sys.argv[6]

    SEQ = False

    cuda = "./output/TIMES/cuda/"
    cuda_sm = "./output/TIMES/cuda_sm/"
    seq = "./output/TIMES/seq/"
    
    cuda_files = os.listdir(cuda)
    cuda_sm_files = os.listdir(cuda_sm)
    seq_files = os.listdir(seq)
    
    cuda_files = np.sort(cuda_files)
    cuda_sm_files = np.sort(cuda_sm_files)
    seq_files = np.sort(seq_files)

    file_names = []

    total_cuda = []
    kernel_cuda = []
    allocate_cuda = []
    copy_cuda = []
    total_cuda2 = []
    kernel_cuda2 = []
    allocate_cuda2 = []
    copy_cuda2 = []
    total_cuda3 = []
    kernel_cuda3 = []
    allocate_cuda3 = []
    copy_cuda3 = []
    for f in cuda_files:
        tmp = f.split(".")[0].split("_")
        file_names.append(tmp[2] + "_" + tmp[3])
        if f.startswith("K" + str(K) + "_iters" + str(iters) + "_"):
            with open(cuda + f) as file:
                total_cuda.append(float(file.readline().replace("\n", "")))
                kernel_cuda.append(float(file.readline().replace("\n", "")))
                allocate_cuda.append(float(file.readline().replace("\n", "")))
                copy_cuda.append(float(file.readline().replace("\n", "")))
        if f.startswith("K" + str(K2) + "_iters" + str(iters2) + "_"):
            with open(cuda + f) as file:
                total_cuda2.append(float(file.readline().replace("\n", "")))
                kernel_cuda2.append(float(file.readline().replace("\n", "")))
                allocate_cuda2.append(float(file.readline().replace("\n", "")))
                copy_cuda2.append(float(file.readline().replace("\n", "")))
        if f.startswith("K" + str(K3) + "_iters" + str(iters3) + "_"):
            with open(cuda + f) as file:
                total_cuda3.append(float(file.readline().replace("\n", "")))
                kernel_cuda3.append(float(file.readline().replace("\n", "")))
                allocate_cuda3.append(float(file.readline().replace("\n", "")))
                copy_cuda3.append(float(file.readline().replace("\n", "")))

    total_cuda_sm = []
    kernel_cuda_sm = []
    allocate_cuda_sm = []
    copy_cuda_sm = []
    total_cuda_sm2 = []
    kernel_cuda_sm2 = []
    allocate_cuda_sm2 = []
    copy_cuda_sm2 = []
    total_cuda_sm3 = []
    kernel_cuda_sm3 = []
    allocate_cuda_sm3 = []
    copy_cuda_sm3 = []
    for f in cuda_sm_files:
        if f.startswith("K" + str(K) + "_iters" + str(iters) + "_"):
            with open(cuda_sm + f) as file:
                total_cuda_sm.append(float(file.readline().replace("\n", "")))
                kernel_cuda_sm.append(float(file.readline().replace("\n", "")))
                allocate_cuda_sm.append(float(file.readline().replace("\n", "")))
                copy_cuda_sm.append(float(file.readline().replace("\n", "")))

        if f.startswith("K" + str(K2) + "_iters" + str(iters2) + "_"):
            with open(cuda_sm + f) as file:
                total_cuda_sm2.append(float(file.readline().replace("\n", "")))
                kernel_cuda_sm2.append(float(file.readline().replace("\n", "")))
                allocate_cuda_sm2.append(float(file.readline().replace("\n", "")))
                copy_cuda_sm2.append(float(file.readline().replace("\n", "")))
        if f.startswith("K" + str(K3) + "_iters" + str(iters3) + "_"):
            with open(cuda_sm + f) as file:
                total_cuda_sm3.append(float(file.readline().replace("\n", "")))
                kernel_cuda_sm3.append(float(file.readline().replace("\n", "")))
                allocate_cuda_sm3.append(float(file.readline().replace("\n", "")))
                copy_cuda_sm3.append(float(file.readline().replace("\n", "")))

    # speedup = []
    # for i in range(len(kernel_cuda)):
    #     speedup.append(kernel_cuda[i] / kernel_cuda_sm[i])
 
    fig, ax = plt.subplots()

    bar_width = 0.1

    ax.bar(np.arange(len(kernel_cuda)), kernel_cuda, bar_width, label="cuda " + str(K) + " " + str(iters))
    ax.bar(np.arange(len(kernel_cuda2)) + bar_width, kernel_cuda2, bar_width, label="cuda " + str(K2) + " " + str(iters2))
    ax.bar(np.arange(len(kernel_cuda3)) + bar_width + bar_width, kernel_cuda3, bar_width, label="cuda " + str(K3) + " " + str(iters3))
    ax.bar(np.arange(len(kernel_cuda_sm)) + bar_width + bar_width + bar_width, kernel_cuda_sm, bar_width, label="cuda_sm " + str(K) + " " + str(iters))
    ax.bar(np.arange(len(kernel_cuda_sm2)) + bar_width + bar_width + bar_width + bar_width, kernel_cuda_sm2, bar_width, label="cuda_sm " + str(K2) + " " + str(iters2))
    ax.bar(np.arange(len(kernel_cuda_sm3)) + bar_width + bar_width + bar_width + bar_width + bar_width, kernel_cuda_sm3, bar_width, label="cuda_sm " + str(K3) + " " + str(iters3))


    # ax2 = ax.twinx()
    # ax2.set_ylabel("Speedup", color="b")
    # ax2.tick_params("y", colors="b")
    # ax2.plot(speedup, color="b")

    ax.set_xlabel('File')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(np.arange(len(kernel_cuda)) + bar_width + bar_width + bar_width / 2)
    ax.set_xticklabels(file_names)
    ax.legend()

    fig.tight_layout()
    plt.show()
    