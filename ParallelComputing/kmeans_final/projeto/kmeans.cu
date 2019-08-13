
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <math.h>
#include <float.h>
#include "timer.h"
#include "lodepng.h"

#define CHECK_ERROR(x) {if((x) != cudaSuccess){ printf("CUDA error at %s:%d\n", __FILE__, __LINE__); printf("%s\n", cudaGetErrorString(cudaGetLastError())); exit(EXIT_FAILURE);}}

// Struct to store the data of each pixel of the image. Used by k-means to compute the distance.
struct Pixel { int r = 0; int g = 0; int b = 0; int a = 0; int x = 0; int y = 0; };
// Load the image using lodepng library.
void load_image(const char* file_name, unsigned char** image, unsigned int* width, unsigned int* height);
// Save the image using lodepng library.
void save_image(const char* file_name, const unsigned char* image, unsigned width, unsigned height);
// Euclidean distance.
__device__ __host__ double squared_distance(const Pixel& first, const Pixel& second);
// Average colors of each cluster.
__global__ void kernel_add_new_colors(const Pixel* means, Pixel* new_colors, const int data_size, const int* assign); 
// Average color of each pixel in the cluster.
__global__ void kernel_compute_means(Pixel* means, const Pixel* sum, const int* counts);
// Compute the nearest pixel for each cluster and assign it to the cluster.
__global__ void kernel_compute_clusters(const Pixel* data, const int data_size, const Pixel* means, Pixel* sum, int* assign, int* counts, const int K);
// Sequential k-means on CPU.
void k_means(Pixel* data, Pixel* new_colors, Pixel* means, const int data_size, const int k, const int iterations);
// K-Means using CUDA SM
__global__ void kernel_sm_compute_clusters(const Pixel* data, const int data_size, const Pixel* means, Pixel* sum, int* assign, int* counts, const int K);


int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: input_image.png K iterations cuda|cuda_sm|seq" << "\n";
		std::cout << "Example CUDA: input_image.png 200 100 cuda" << "\n";
		std::cout << "Example CUDA: input_image.png 200 100 cuda_sm" << "\n";
        std::cout << "Example Sequential: input_image.png 200 100 seq" << "\n";
        return -1;
    }
    const std::string file_name = argv[1];
    const int K = atoi(argv[2]);
    const int iters = atoi(argv[3]);
    const std::string type = argv[4];
	std::string tmp = file_name;
	std::string slash = "input/";
	tmp.replace(tmp.find(slash), slash.size(), "");

    const std::string file_name_output = "output/out_" + type + "_K" + argv[2] + "_iters" + argv[3] + "_" + tmp;

    const int loop = 32; // Amount of tests.
    double time_kernel = 0; // Timer.
    double time_allocate = 0; // Timer.
    double time_copy = 0; // Timer.
    double time_total = 0.0;

    unsigned char* input_image;
    unsigned char* output_image;
    unsigned int width, height;
    load_image(file_name.c_str(), &input_image, &width, &height);

    const int data_size = width * height;
    const int pixel_size = sizeof(Pixel);
    Pixel* data_array = new Pixel[data_size]; // Array with all the data of the image.
    Pixel* new_colors = new Pixel[data_size]; // Average RGBA colors computed by k-means.
    Pixel* km = new Pixel[K]; // Center of each clusters.
    
    // Copy the image to data_array.
    int i_d = 0;
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            const int offset = ((x * width) + y) * 4;
            int r = (int)input_image[offset];
            int g = (int)input_image[offset + 1];
            int b = (int)input_image[offset + 2];
            int a = (int)input_image[offset + 3];
            data_array[i_d].r = r;
            data_array[i_d].g = g;
            data_array[i_d].b = b;
            data_array[i_d].a = a;
            data_array[i_d].x = x;
            data_array[i_d].y = y;
            i_d++;
        }
    }

    // Initial position of the clusters, random.
    static std::random_device seed;
    static std::mt19937 rng(seed());
    std::uniform_int_distribution<int> indices(0, data_size - 1);
    for (int i = 0; i < K; i++) {
        km[i] = data_array[indices(rng)];
    }

    // To copy the x and y of each pixel.
    memcpy(new_colors, data_array, data_size * sizeof(Pixel));

    // Sequential k-means.
    if (type == "seq") {
        std::cout << "[CPU] K-Means started..." << "\n";
        StartTimer();
        for (int i = 0; i < loop; i++) {
            k_means(data_array, new_colors, km, data_size, K, iters);
        }
        time_kernel = GetTimer() / loop;
        time_total += time_kernel;
        std::cout << "[CPU] K-Means done." << "\n";
        std::cout << "[CPU] Time K-Means: " << time_kernel << " milliseconds.\n";
    }
    
    Pixel* gpu_data;
    Pixel* gpu_sum;
    Pixel* gpu_means;
    int* gpu_counts;
    Pixel* cpu_data;
    Pixel* cpu_means;
    int* gpu_assign;
    Pixel* gpu_new_colors;
    Pixel* cpu_new_colors;

	cpu_data = data_array;
	cpu_means = km;
	cpu_new_colors = new_colors;
	const int ths = 512;
	const int block = (data_size + ths) / ths;
	std::cout << "Threads: " << ths << "\n";
	std::cout << "Blocks: " << block << "\n";
    // Cuda k-means.
    if (type == "cuda") {
      
        // GPU memory allocation.
        std::cout << "[CUDA] Allocating memory..." << "\n";
        StartTimer();
        CHECK_ERROR(cudaMalloc((void **)&gpu_data, data_size * pixel_size));
        CHECK_ERROR(cudaMalloc((void **)&gpu_sum, K * pixel_size));
        CHECK_ERROR(cudaMalloc((void **)&gpu_means, K * pixel_size));
        CHECK_ERROR(cudaMalloc((void **)&gpu_counts, K * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void **)&gpu_assign, data_size * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void **)&gpu_new_colors, data_size * pixel_size));
        time_allocate = GetTimer();
        time_total += time_allocate;
        std::cout << "[CUDA] Done allocating memory." << "\n";
        std::cout << "[CUDA] Time allocating memory: " << time_allocate << " milliseconds.\n";

        // Copy to GPU memory.
        std::cout << "[CUDA] Copying to GPU..." << "\n";
        StartTimer();
        CHECK_ERROR(cudaMemcpy(gpu_data, cpu_data,  data_size * pixel_size, cudaMemcpyHostToDevice));
        CHECK_ERROR(cudaMemcpy(gpu_means, cpu_means,  K * pixel_size, cudaMemcpyHostToDevice));
        CHECK_ERROR(cudaMemcpy(gpu_new_colors, cpu_new_colors,  data_size * pixel_size, cudaMemcpyHostToDevice));
        time_copy = GetTimer();
        time_total += time_copy;
        std::cout << "[CUDA] Done copying to GPU." << "\n";
        std::cout << "[CUDA] Time copying to GPU: " << time_copy << " milliseconds.\n";

        // Run GPU kernels.
        std::cout << "[CUDA] K-Means started..." << "\n";
        cudaDeviceSynchronize();
        StartTimer();
        for (int i = 0; i < loop; i++) {
            for (int iteration = 0; iteration < iters; iteration++) {
                cudaMemset(gpu_counts, 0, K * sizeof(int));
                cudaMemset(gpu_sum, 0, K * pixel_size);
                kernel_compute_clusters<<<block, ths>>>(gpu_data, data_size, gpu_means, gpu_sum, gpu_assign, gpu_counts, K);
                cudaDeviceSynchronize();
                kernel_compute_means<<<1, K>>>(gpu_means, gpu_sum, gpu_counts);
                cudaDeviceSynchronize();
            }
            kernel_add_new_colors<<<block, ths>>>(gpu_means, gpu_new_colors, data_size, gpu_assign);
        }
        cudaDeviceSynchronize();
        time_kernel = GetTimer() / loop;
        time_total += time_kernel;
        std::cout << "[CUDA] K-Means done." << "\n";
        std::cout << "[CUDA] Time K-Means (mean time): " << time_kernel << " milliseconds.\n";

        // Copy to CPU memory.
        std::cout << "[CUDA] Copying to CPU..." << "\n";
        StartTimer();
        CHECK_ERROR(cudaMemcpy(cpu_means, gpu_means,  K * pixel_size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(cpu_new_colors, gpu_new_colors,  data_size * pixel_size, cudaMemcpyDeviceToHost));
        time_copy += GetTimer();
        time_total += time_copy;
        std::cout << "[CUDA] Done copying to CPU." << "\n";
        std::cout << "[CUDA] Time copying to CPU: " << time_copy << " milliseconds.\n";
		km = cpu_means;
        new_colors = cpu_new_colors;
    } else if (type == "cuda_sm") {
		// GPU memory allocation.
        std::cout << "[CUDA SM] Allocating memory..." << "\n";
        StartTimer();
        CHECK_ERROR(cudaMalloc((void **)&gpu_data, data_size * pixel_size));
        CHECK_ERROR(cudaMalloc((void **)&gpu_sum, K * pixel_size));
        CHECK_ERROR(cudaMalloc((void **)&gpu_means, K * pixel_size));
        CHECK_ERROR(cudaMalloc((void **)&gpu_counts, K * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void **)&gpu_assign, data_size * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void **)&gpu_new_colors, data_size * pixel_size));
        time_allocate = GetTimer();
        time_total += time_allocate;
        std::cout << "[CUDA SM] Done allocating memory." << "\n";
        std::cout << "[CUDA SM] Time allocating memory: " << time_allocate << " milliseconds.\n";

        // Copy to GPU memory.
        std::cout << "[CUDA SM] Copying to GPU..." << "\n";
        StartTimer();
        CHECK_ERROR(cudaMemcpy(gpu_data, cpu_data,  data_size * pixel_size, cudaMemcpyHostToDevice));
        CHECK_ERROR(cudaMemcpy(gpu_means, cpu_means,  K * pixel_size, cudaMemcpyHostToDevice));
        CHECK_ERROR(cudaMemcpy(gpu_new_colors, cpu_new_colors,  data_size * pixel_size, cudaMemcpyHostToDevice));
        time_copy = GetTimer();
        time_total += time_copy;
        std::cout << "[CUDA SM] Done copying to GPU." << "\n";
        std::cout << "[CUDA SM] Time copying to GPU: " << time_copy << " milliseconds.\n";


		const int shared_memory = K * pixel_size;

        // Run GPU kernels.
        std::cout << "[CUDA SM] K-Means started..." << "\n";
        cudaDeviceSynchronize();
        StartTimer();
        for (int i = 0; i < loop; i++) {
			for (int iteration = 0; iteration < iters; iteration++) {
				cudaMemset(gpu_counts, 0, K * sizeof(int));
				cudaMemset(gpu_sum, 0, K * pixel_size);
				kernel_sm_compute_clusters << <block, ths, shared_memory >> > (gpu_data, data_size, gpu_means, gpu_sum, gpu_assign, gpu_counts, K);
				cudaDeviceSynchronize();
				kernel_compute_means << <1, K >> > (gpu_means, gpu_sum, gpu_counts);
				cudaDeviceSynchronize();
			}
			kernel_add_new_colors <<<block, ths>> > (gpu_means, gpu_new_colors, data_size, gpu_assign);
        }
        cudaDeviceSynchronize();
        time_kernel = GetTimer() / loop;
        time_total += time_kernel;
        std::cout << "[CUDA SM] K-Means done." << "\n";
        std::cout << "[CUDA SM] Time K-Means (mean time): " << time_kernel << " milliseconds.\n";

        // Copy to CPU memory.
        std::cout << "[CUDA SM] Copying to CPU..." << "\n";
        StartTimer();
        CHECK_ERROR(cudaMemcpy(cpu_means, gpu_means,  K * pixel_size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(cpu_new_colors, gpu_new_colors,  data_size * pixel_size, cudaMemcpyDeviceToHost));
        time_copy += GetTimer();
        time_total += time_copy;
        std::cout << "[CUDA SM] Done copying to CPU." << "\n";
        std::cout << "[CUDA SM] Time copying to CPU: " << time_copy << " milliseconds.\n";
		km = cpu_means;
        new_colors = cpu_new_colors;
    }

    data_array = new_colors;

    // Draw a small + sign on the center of the clusters.
    const int r = 3;
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < data_size; i++) {
            if (km[j].x == data_array[i].x && km[j].y == data_array[i].y) {
                data_array[i].r = 0;
                data_array[i].g = 0;
                data_array[i].b = 0;
                data_array[i].a = 0;
				for (int h = 0; h < r; h++) {
                    if (i + h < data_size) {
                        data_array[i + h].r = 0;
                        data_array[i + h].g = 0;
                        data_array[i + h].b = 0;
                        data_array[i + h].a = 255;
                        if (i - width * h < data_size) {
                            data_array[i - width * h].r = 0;
                            data_array[i - width * h].g = 0;
                            data_array[i - width * h].b = 0;
                            data_array[i - width * h].a = 255;
                        }
                    }
                    if (i - h > 0) {
                        data_array[i - h].r = 0;
                        data_array[i - h].g = 0;
                        data_array[i - h].b = 0;
                        data_array[i - h].a = 255;
                        if (i + width * h > 0) {
                            data_array[i + width * h].r = 0;
                            data_array[i + width * h].g = 0;
                            data_array[i + width * h].b = 0;
                            data_array[i + width * h].a = 255;
                        }
                    }
				}
            }
        }
    }

    // Copy data_array to output_image
    output_image = new unsigned char[(width * height) * sizeof(unsigned char) * 4];
	i_d = 0;
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
			if (i_d < data_size && data_array[i_d].x == x && data_array[i_d].y == y){
                int offset = ((x * width) + y) * 4;
                output_image[offset] = (unsigned char)data_array[i_d].r;
                output_image[offset + 1] = (unsigned char)data_array[i_d].g;
                output_image[offset + 2] = (unsigned char)data_array[i_d].b;
                output_image[offset + 3] = (unsigned char)data_array[i_d].a;
                i_d++;
            }
		}
	}
    
    save_image(file_name_output.c_str(), output_image, width, height);
    
    if (type == "cuda" || type == "cuda_sm") {
        //std::cout << "[CUDA] Deallocating GPU memory..." << "\n";
        cudaFree(gpu_data);
        cudaFree(gpu_sum);
        cudaFree(gpu_means);
        cudaFree(gpu_counts);
        cudaFree(gpu_assign);
        cudaFree(gpu_new_colors);
        delete[] cpu_data;
        delete[] cpu_means;
        delete[] cpu_new_colors;
        //std::cout << "[CUDA] Done deallocating GPU memory." << "\n";
    }
    //std::cout << "Deallocating CPU memory..." << "\n";
    delete[] input_image;
    delete[] output_image;
    //delete data_array;
    //delete new_colors;
    //delete km;
    //std::cout << "Done deallocating CPU memory." << "\n";

    std::cout << "Total time: " << time_total << " milliseconds.\n";
    std::cout << "Kernel time: " << time_kernel << " milliseconds.\n";
    std::cout << "Allocate time: " << time_allocate << " milliseconds.\n";
    std::cout << "Copy time: " << time_copy << " milliseconds.\n";

    const std::string times_output = "output/TIMES/" + type + "/" + "K" + argv[2] + "_iters" + argv[3] + "_" + tmp + ".txt";
	std::cout << "Saving times to: " << times_output << "\n";
    FILE *f = fopen(times_output.c_str(), "w");
    if (f == NULL) {
        printf("Error while saving times.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(f, "%f\n", time_total);
    fprintf(f, "%f\n", time_kernel);
    fprintf(f, "%f\n", time_allocate);
    fprintf(f, "%f\n", time_copy);
    fclose(f);

    return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------//

// Common functions

// Load the image using lodepng library.
void load_image(const char* file_name, unsigned char** image, unsigned int* width, unsigned int* height) {
    std::cout << "Loading: " << file_name << "\n";
    unsigned int error = lodepng_decode32_file(image, width, height, file_name);
    if (error) { std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl; exit(-1); }
    std::cout << "Width: " << (int)*width << "\n";
    std::cout << "Height: " << (int)*height << "\n";
}

// Save the image using lodepng library.
void save_image(const char* file_name, const unsigned char* image, unsigned width, unsigned height) {
    unsigned error = lodepng_encode32_file(file_name, image, width, height);
    if (error) { std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl; exit(-1);}
    std::cout << "Image saved: " << file_name << "\n";
}

// Euclidean distance.
__device__ __host__ double squared_distance(const Pixel& first, const Pixel& second) {
    return ((first.r - second.r) * (first.r - second.r)) + ((first.g - second.g) * (first.g - second.g)) + ((first.b - second.b) * (first.b - second.b)) + ((first.a - second.a) * (first.a - second.a)) +((first.x - second.x) * (first.x - second.x)) + ((first.y - second.y) * (first.y - second.y));
}

// CUDA

// Average colors of each cluster.
__global__ void kernel_add_new_colors(const Pixel* means, Pixel* new_colors, const int data_size, const int* assign) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < data_size) {
        const int cluster = assign[index];
		new_colors[index].r = means[cluster].r;
		new_colors[index].g = means[cluster].g;
		new_colors[index].b = means[cluster].b;
		new_colors[index].a = means[cluster].a;
    }
}

// Average color of each pixel in the cluster.
__global__ void kernel_compute_means(Pixel* means, const Pixel* sum, const int* counts) {
    const int cluster = threadIdx.x;
    const int count = counts[cluster] == 0 ? 1 : counts[cluster];
    means[cluster].r = sum[cluster].r / count;
    means[cluster].g = sum[cluster].g / count;
    means[cluster].b = sum[cluster].b / count;
    means[cluster].a = sum[cluster].a / count;
    means[cluster].x = sum[cluster].x / count;
    means[cluster].y = sum[cluster].y / count;
}

// Compute the nearest pixel for each cluster and assign it to the cluster.
__global__ void kernel_compute_clusters(const Pixel* data, const int data_size, const Pixel* means, Pixel* sum, int* assign, int* counts, const int K) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < data_size) {
        const Pixel p = data[index];
        double best_distance = DBL_MAX;
        int best_cluster = 0;
        for (int cluster = 0; cluster < K; cluster++) {
            const double distance = squared_distance(p, means[cluster]);
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster;
            }
        }
        assign[index] = best_cluster;
        atomicAdd(&sum[best_cluster].r, p.r);
        atomicAdd(&sum[best_cluster].g, p.g);
        atomicAdd(&sum[best_cluster].b, p.b);
        atomicAdd(&sum[best_cluster].a, p.a);
        atomicAdd(&sum[best_cluster].x, p.x);
        atomicAdd(&sum[best_cluster].y, p.y);
        atomicAdd(&counts[best_cluster], 1);
    }
}
// End CUDA

// CUDA SM
// Version using shared memory.
__global__ void kernel_sm_compute_clusters(const Pixel* data, const int data_size, const Pixel* means, Pixel* sum, int* assign, int* counts, const int K) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ Pixel s_means[];
	if (threadIdx.x < K) {
		s_means[threadIdx.x] = means[threadIdx.x];
	}
	__syncthreads();
	if (index < data_size) {
		const Pixel pix = data[index];
		double best_distance = DBL_MAX;
		int best_cluster = 0;
		for (int cluster = 0; cluster < K; cluster++) {
			const double distance = squared_distance(pix, s_means[cluster]);
			if (distance < best_distance) {
				best_distance = distance;
				best_cluster = cluster;
			}
		}
		assign[index] = best_cluster;
		atomicAdd(&sum[best_cluster].r, pix.r);
		atomicAdd(&sum[best_cluster].g, pix.g);
		atomicAdd(&sum[best_cluster].b, pix.b);
		atomicAdd(&sum[best_cluster].a, pix.a);
		atomicAdd(&sum[best_cluster].x, pix.x);
		atomicAdd(&sum[best_cluster].y, pix.y);
		atomicAdd(&counts[best_cluster], 1);
	}
}
// End CUDA sm

// CPU

// Sequential k-means on CPU.
void k_means(Pixel* data, Pixel* new_colors, Pixel* means, const int data_size, const int k, const int iterations) {
	int* assign = new int[data_size];
    for (int iteration = 0; iteration < iterations; iteration++) {
        for (int i = 0; i < data_size; i++) {
            double best_distance = DBL_MAX;
            int best_cluster = 0;
            for (int cluster = 0; cluster < k; cluster++) {
                const double distance = squared_distance(data[i], means[cluster]);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }
            assign[i] = best_cluster;
        }

        Pixel* new_means = new Pixel[k];
        int* counts = new int[k];
		memset(counts, 0, k * sizeof(int));
        for (int i = 0; i < data_size; i++) {
            const int cluster = assign[i];
            new_means[cluster].r += data[i].r;
            new_means[cluster].g += data[i].g;
            new_means[cluster].b += data[i].b;
            new_means[cluster].a += data[i].a;
            new_means[cluster].x += data[i].x;
            new_means[cluster].y += data[i].y;
            counts[cluster] += 1;
        }

        for (int cluster = 0; cluster < k; cluster++) {
            const int count = counts[cluster] == 0 ? 1 : counts[cluster];
            means[cluster].r = new_means[cluster].r / count;
            means[cluster].g = new_means[cluster].g / count;
            means[cluster].b = new_means[cluster].b / count;
            means[cluster].a = new_means[cluster].a / count;
            means[cluster].x = new_means[cluster].x / count;
            means[cluster].y = new_means[cluster].y / count;
        }
        delete[] new_means;
        delete[] counts;
    }
    for (int i = 0; i < data_size; i++) {
        const int cluster = assign[i];
        new_colors[i].r = means[cluster].r;
        new_colors[i].g = means[cluster].g;
        new_colors[i].b = means[cluster].b;
        new_colors[i].a = means[cluster].a;
    }
    delete[] assign;
}
// End CPU
