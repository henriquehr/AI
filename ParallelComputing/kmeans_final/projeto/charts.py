import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
 
    K = sys.argv[1]
    iters = sys.argv[2]

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
    for f in cuda_files:
        if f.startswith("K" + str(K) + "_iters" + str(iters) + "_"):
            tmp = f.split(".")[0].split("_")
            file_names.append(tmp[2] + "_" + tmp[3])
            with open(cuda + f) as file:
                total_cuda.append(float(file.readline().replace("\n", "")))
                kernel_cuda.append(float(file.readline().replace("\n", "")))
                allocate_cuda.append(float(file.readline().replace("\n", "")))
                copy_cuda.append(float(file.readline().replace("\n", "")))

    total_cuda_sm = []
    kernel_cuda_sm = []
    allocate_cuda_sm = []
    copy_cuda_sm = []
    for f in cuda_sm_files:
        if f.startswith("K" + str(K) + "_iters" + str(iters) + "_"):
            with open(cuda_sm + f) as file:
                total_cuda_sm.append(float(file.readline().replace("\n", "")))
                kernel_cuda_sm.append(float(file.readline().replace("\n", "")))
                allocate_cuda_sm.append(float(file.readline().replace("\n", "")))
                copy_cuda_sm.append(float(file.readline().replace("\n", "")))

    speedup = []
    for i in range(len(kernel_cuda)):
        speedup.append(kernel_cuda[i] / kernel_cuda_sm[i])

    if SEQ:
        total_seq = []
        kernel_seq = []
        allocate_seq = []
        copy_seq = []
        for f in seq_files:
            if f.startswith("K" + str(K) + "_iters" + str(iters) + "_"):
                with open(seq + f) as file:
                    total_seq.append(float(file.readline().replace("\n", "")))
                    kernel_seq.append(float(file.readline().replace("\n", "")))
                    allocate_seq.append(float(file.readline().replace("\n", "")))
                    copy_seq.append(float(file.readline().replace("\n", "")))

 
    fig, ax = plt.subplots()

    bar_width = 0.2

    ax.bar(np.arange(len(kernel_cuda)), kernel_cuda, bar_width, label="cuda")

    ax.bar(np.arange(len(kernel_cuda_sm)) + bar_width, kernel_cuda_sm, bar_width, label="cuda_sm")
    if SEQ:
        ax.bar(np.arange(len(kernel_seq)) + bar_width + bar_width, kernel_seq, bar_width, label="sequential")

    ax2 = ax.twinx()
    ax2.set_ylabel("Speedup", color="g")
    ax2.tick_params("y", colors="g")
    ax2.plot(speedup, color="g")

    ax.set_xlabel('File')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(np.arange(len(kernel_cuda)) + bar_width / 2)
    ax.set_xticklabels(file_names)
    ax.legend()

    fig.tight_layout()
    plt.show()
    