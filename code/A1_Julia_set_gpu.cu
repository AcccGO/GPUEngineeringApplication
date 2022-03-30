#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../cuda_by_example-source-code/cuda_by_example/common/GL/glut.h"
#include "../cuda_by_example-source-code/cuda_by_example/common/gl_helper.h"
#include "../cuda_by_example-source-code/cuda_by_example/common/cpu_bitmap.h"
#include "../cuda_by_example-source-code/cuda_by_example/common/book.h"
#include <stdio.h>

using namespace std;
#define DIM 1000
#define PI 3.14159265

struct  cuComplex {
	float r;
	float i;
	__device__ cuComplex(float a, float b) {
		r = a;
		i = b;
	}
	__device__ float magnitude2(void) {
		return r * r + i * i;
	}

	__device__ cuComplex operator * (const cuComplex& a) {
		return cuComplex(r*a.r - i * a.i, i*a.r + r * a.i);
	}

	__device__ cuComplex operator + (const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__device__ int julia2(int x, int y) {
	const float scale = 2.0;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(0.285, 0.01);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 2)
			return i;
	}
	return 0;
}

__device__ int julia3(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.576,0.456);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 2)
			return i;
	}
	return 0;
}

__global__ void kernel(unsigned char *ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	int juliaValue = julia2(x, y);

	ptr[offset * 4 + 0] = 127 * (__sinf(juliaValue)+1);
	ptr[offset * 4 + 1] = 127 * (__sinf(juliaValue+2.0/3*PI) + 1);
	ptr[offset * 4 + 2] = 127 * (__sinf(juliaValue+4.0/3*PI) + 1);
	ptr[offset * 4 + 3] = 255;
	
}
struct DataBlock {
	unsigned char   *dev_bitmap;
};
int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	cudaEvent_t     start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	dim3 grid(DIM, DIM);
	kernel << <grid, 1 >> > (dev_bitmap);

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time to generate:  %3.1f ms\n", elapsedTime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	HANDLE_ERROR(cudaFree(dev_bitmap));
}
