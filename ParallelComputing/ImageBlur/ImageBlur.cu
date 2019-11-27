// v0.2 modified by WZ

#include <fstream>

//#include <wb.h>
//#include "/home/prof/wagner/ci853/labs/wb4.h" // use our lib instead (under construction)
//#include "/home/wagner/ci853/labs-achel/wb.h" // use our lib instead (under construction)

#include "wb4.h" // use our lib instead (under construction)

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLUR_SIZE 5 
#define TILE_SIZE 16 // Only used to compare with the shared memory version
#define BLOCK_SIZE (TILE_SIZE + TILE_SIZE) // Only used to compare with the shared memory version

//@@ INSERT CODE HERE
__global__ void blurKernel(const unsigned char* input, unsigned char* output, const int w, const int h) {
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	if (Col < w && Row < h) {
		int pix_red = 0;
		int pix_green = 0;
		int pix_blue = 0;
		int pixels = 0;
		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;
				if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
					int offset = (curRow * w + curCol) * 3;
					pix_red += input[offset];
					pix_green += input[offset + 1];
					pix_blue += input[offset + 2];
					pixels++;
				}
			}
		}
		output[(Row * w + Col) * 3] = (unsigned char)(pix_red / pixels);
		output[((Row * w + Col) * 3) + 1] = (unsigned char)(pix_green / pixels);
		output[((Row * w + Col) * 3) + 2] = (unsigned char)(pix_blue / pixels);
	}
}


int main(int argc, char *argv[]) {
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	unsigned char *hostInputImageData;
	unsigned char *hostOutputImageData;
	unsigned char *deviceInputImageData;
	unsigned char *deviceOutputImageData;
	double times;

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 1);
	printf("imagem de entrada: %s\n", inputImageFile);

	//  inputImage = wbImportImage(inputImageFile);
	inputImage = wbImport(inputImageFile);

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);

	// NOW: input and output images are RGB (3 channel)
	outputImage = wbImage_new(imageWidth, imageHeight, 3);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * sizeof(unsigned char) * 3);
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(unsigned char) * 3);
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	//int ths = 64;
	//dim3 dimBlock(imageWidth / ths, imageHeight / ths);
	//dim3 dimGrid((imageWidth / dimBlock.x) + 1, (imageHeight / dimBlock.y) + 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((imageWidth / TILE_SIZE) + 1, (imageHeight / TILE_SIZE) + 1);
	printf("dimBlock x: %d | y: %d | z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	printf("dimGrid x: %d | y: %d | z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	cudaDeviceSynchronize();
	for (int i = 0; i < 32; i++) {
		blurKernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
	}
	cudaDeviceSynchronize();
    times = wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost);

	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);
	// DEBUG: if you want to see your image, 
	//   will generate file bellow in current directory
	//wbExport("blurred.ppm", outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

    std::ofstream out;
    out.open("times.txt", std::ios::app);
	out << times << "\n";
    out.close();
	printf(">>>Time: %f\n", times);
	return 0;
}
