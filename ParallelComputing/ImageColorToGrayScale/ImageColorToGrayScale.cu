//
//   v0.2 corrigida por WZola aug/2017 para ficar de acordo com novo wb.h 
//        (ou seja de acordo com wb4.h)
//        

//#include <wb.h>     // original
//#include "/home/prof/wagner/ci853/labs/wb.h" // use our lib instead (under construction)
//#include "/home/wagner/ci853/labs-achel/wb.h" // use our lib instead (under construction)
//#include "/home/ci853/wb4.h"   // wb4.h on gp1 machine
#include "wb4.h" // use our new lib, wherever it is

#include <string.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE
__global__ void kernel_to_gray_scale(unsigned char* input, unsigned char* output, int w, int h) {
	int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx_x < w && idx_y < h) {
		int idx = (idx_y * w) + idx_x;
		int rgbOffset = idx * 3;
		unsigned char r = input[rgbOffset];
		unsigned char g = input[rgbOffset + 1];
		unsigned char b = input[rgbOffset + 2];
		output[idx] = 0.21f * r + 0.71f * g + 0.07f * b;
	}
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;

	//  float *hostInputImageData;
	//  float *hostOutputImageData;
	//  float *deviceInputImageData;
	//  float *deviceOutputImageData;

	unsigned char *hostInputImageData;
	unsigned char *hostOutputImageData;
	unsigned char *deviceInputImageData;
	unsigned char *deviceOutputImageData;

	args = wbArg_read(argc, argv); /* parse the input arguments */
  //  show_args( args ); // debug

  //  inputImageFile = wbArg_getInputFileName(args, 2);
	inputImageFile = argv[2];

	//  inputImage = wbImportImage(inputImageFile);
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	int blocks = 64;
	dim3 dimBlock(imageWidth / blocks, imageHeight / blocks);
	dim3 dimGrid((imageWidth / dimBlock.x) + 1, (imageHeight / dimBlock.y) + 1);
	printf("dimBlock x: %d | y: %d | z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	printf("dimGrid x: %d | y: %d | z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

	//////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	cudaDeviceSynchronize();
	for (int i = 0; i < 32; i++) {
		kernel_to_gray_scale<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
	}
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");
	
	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
	
	//wbExport("gray.ppm", outputImage);

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}