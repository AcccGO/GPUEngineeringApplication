/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/


#include "../cuda_by_example-source-code/cuda_by_example/common/book.h"
#include "../cuda_by_example-source-code/cuda_by_example/common/cpu_bitmap.h"

#include <math.h>
#include <omp.h>

#define DIM 1000
#define PI 3.14159265

struct cuComplex {
	float   r;
	float   i;
	cuComplex(float a, float b) : r(a), i(b) {}
	float magnitude2(void) { return r * r + i * i; }
	cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i * a.i, i*a.r + r * a.i);
	}
	cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

int julia(int x, int y) {
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

int julia2(int x, int y) {
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

void kernel(unsigned char *ptr) {
	for (int y = 0; y < DIM; y++) {
		for (int x = 0; x < DIM; x++) {
			int offset = x + y * DIM;

			int juliaValue = julia2(x, y);
			ptr[offset * 4 + 0] = 127 * (sin(juliaValue) + 1);
			ptr[offset * 4 + 1] = 127 * (sin(juliaValue + 2.0 / 3 * PI) + 1);
			ptr[offset * 4 + 2] = 127 * (sin(juliaValue + 4.0 / 3 * PI) + 1);
			ptr[offset * 4 + 3] = 255;
		}
	}
}

int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *ptr = bitmap.get_ptr();

	double start = omp_get_wtime();

	kernel(ptr);

	double end = omp_get_wtime();
	double duration = end - start;
	printf("%3.1f ms\n", duration * 1000);

	bitmap.display_and_exit();
}

