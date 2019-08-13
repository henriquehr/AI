/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */
#include "../utils/chrono.c"
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

//#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    int first = 500000;
    int numElements = first;
    int tests = 5;
    long long *all_times = (long long *)malloc(tests * sizeof(long long));
    if (all_times == NULL) {
        fprintf(stderr, "Falha ao alocar vetor de tempos\n");
        exit(EXIT_FAILURE);
    }
    for (int all = 0; all < tests; all++){
        cudaError_t err = cudaSuccess;
        size_t size = numElements * sizeof(float);
        printf("[Vector addition of %d elements]\n", numElements);

        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C = (float *)malloc(size);
        if (h_A == NULL || h_B == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < numElements; ++i)
        {
            h_A[i] = rand()/(float)RAND_MAX;
            h_B[i] = rand()/(float)RAND_MAX;
        }
        float *d_A = NULL;
        err = cudaMalloc((void **)&d_A, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        float *d_B = NULL;
        err = cudaMalloc((void **)&d_B, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        float *d_C = NULL;
        err = cudaMalloc((void **)&d_C, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        int threadsPerBlock = 256;
        int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
        printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
        timeval t1, t2;
        double elapsedTime;
        //chronometer_t c;
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        //chrono_start(&c);
        for (int i = 0; i < 32; i++) {
            vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        //chrono_stop(&c);
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec);   // us
        double time = elapsedTime;
        //long long time = (chrono_gettotal(&c) / 32) / 1000;
        all_times[all] = time;
        printf(">>>Tamanho do vetor: %d\n", numElements);
        printf(">>>Tempo médio: %lld.\n", time);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < numElements; ++i)
        {
            if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }

        printf("Test PASSED\n");
        err = cudaFree(d_A);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaFree(d_B);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaFree(d_C);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        free(h_A);
        free(h_B);
        free(h_C);
        numElements *= 2;
    	printf("\n");
    }

    FILE *f = fopen("times.csv", "w");
    if (f == NULL)
    {
        printf("Erro ao salvar tempos.\n");
        exit(EXIT_FAILURE);
    }
    printf("\n>>>Tempos:\n");
    numElements = first;
    for (int i = 0; i < tests; i++) {
        printf(">>>Tamanho do vetor: %d ; Tempo: %lld\n", numElements, all_times[i]);
        fprintf(f,"%d;%lld\n", numElements, all_times[i]);
        numElements *= 2;
    }
    fclose(f);
    // free(all_times);
    printf("Done\n");
    return 0;
}
