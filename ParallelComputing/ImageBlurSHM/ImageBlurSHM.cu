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
#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE + TILE_SIZE)
#define SHARED_MEM_SIZE ((BLOCK_SIZE * BLOCK_SIZE) * 3)

//@@ INSERT CODE HERE

__global__ void blurKernelSHM(const unsigned char* input, unsigned char* output, const int w, const int h) {
 	__shared__ unsigned char tile[SHARED_MEM_SIZE]; // array faster than 3D array [][][]
	int x = blockIdx.x * TILE_SIZE + threadIdx.x - BLUR_SIZE;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y - BLUR_SIZE;
	x = max(0, x);
	x = min(x, h - 1);
	y = max(0, y);
	y = min(y, w - 1);
	const int d = (y * w + x) * 3;
	const int offset = (threadIdx.y * BLOCK_SIZE + threadIdx.x) * 3;
	if (offset <= SHARED_MEM_SIZE - 3 || d < (w * h * 3) - 3) {
		tile[offset] = input[d];
		tile[offset + 1] = input[d + 1];
		tile[offset + 2] = input[d + 2];
	}

	__syncthreads();

	if (threadIdx.x >= BLUR_SIZE && threadIdx.x < BLOCK_SIZE - BLUR_SIZE && threadIdx.y >= BLUR_SIZE && threadIdx.y < BLOCK_SIZE - BLUR_SIZE) {
		int pix_red = 0;
		int pix_green = 0;
		int pix_blue = 0;
		int pixels = 0;
		for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; dx++) {
			for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; dy++) {
				int currX = dx + threadIdx.x;
				int currY = dy + threadIdx.y;
				int curr_offset = (currY * BLOCK_SIZE + currX) * 3;
				if (curr_offset < SHARED_MEM_SIZE - 3 && curr_offset >= 0) {
					pix_red += tile[curr_offset];
					pix_green += tile[curr_offset + 1];
					pix_blue += tile[curr_offset + 2];
					pixels++;
				}
			}
		}
		output[d] = (unsigned char)(pix_red / pixels);
		output[d + 1] = (unsigned char)(pix_green / pixels);
		output[d + 2] = (unsigned char)(pix_blue / pixels);
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


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((imageWidth / TILE_SIZE)+1, (imageHeight / TILE_SIZE)+1);
	printf("dimBlock x: %d | y: %d | z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	printf("dimGrid x: %d | y: %d | z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	cudaDeviceSynchronize();
	for (int i = 0; i < 32; i++) {
		blurKernelSHM<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
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
	//wbExport("blurredSM.ppm", outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

    std::ofstream out;
    out.open("timesSHM.txt", std::ios::app);
	out << times << "\n";
    out.close();
	printf(">>>Time: %f\n", times);
	return 0;
}
