//**************************************************************************************
//  Copyright (C) 2019 - 2022, Min Tang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#include "stdafx.h"

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <vector>
#include <iostream>
using namespace std;

#include "vec3f.h"
#include "box.h"
#include "xyz-rgb.h"
#include "timer.h"
#include "bvh.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Check CUDA error.
// API call error handling.
#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
// Check CUDA Runtime status code to accept a specified prompt.
#define CHECK_STATE(msg) checkCudaState(msg, __FILE__, __LINE__)

inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
	if (error != cudaSuccess) {
		std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
		exit(EXIT_FAILURE);
	}
}

inline void checkCudaState(const char *msg, const char *file, const int line)
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "---" << msg << " Error---" << std::endl;
		std::cerr << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
		exit(EXIT_FAILURE);
	}
}

void build_bvh();
void destroy_bvh();
bool findNearest2(xyz2rgb &org, xyz2rgb &ret);
int maxPts();

std::vector<xyz2rgb> gHashTab;
extern t_linearBVH *lTree;

extern "C"
__device__ __host__ bool getColorCUDA(t_linearBVH* d_linear_bvh, vec3f &xyz, float *nearest, vec3f &cr, int &idx, int &pidx);

bool getColor(vec3f &xyz, float *nearest, vec3f &cr, int &idx, int &pidx)
{
	xyz2rgb input(xyz, vec3f(), idx, pidx), ret;

	bool find = findNearest2(input, ret);
	if (!find)
		return false;

	vec3f now = ret.xyz();
	double dist = vdistance(xyz, now);
	if (dist < *nearest) {
		*nearest = dist;
		cr = ret.rgb();
		idx = ret.index();
		pidx = ret.pos();

		return true;
	}
	else
		return false;
}


class SceneData {
protected:
	// Width and height.
	int _cx, _cy;
	// Load xyz position.
	float *_xyzs;
	// Load rgb.
	float *_rgbs;
	
	float *_rgbsNew;
	// AABB bbox.
	BOX _bound;

public:
	void loadRGB(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);
		int sz = _cx*_cy * 3;
		_rgbs = new float[sz];
		fread(_rgbs, sizeof(float), sz, fp);
		fclose(fp);
	}

	void loadXYZ(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);

		int sz = _cx*_cy * 3;
		_xyzs = new float[sz];
		fread(_xyzs, sizeof(float), sz, fp);
		fclose(fp);
	}

	int width() const { return _cx; }
	int height() const { return _cy; }

	virtual void saveAsBmp(float *ptr, char *fn) {
		int sz = _cx * _cy * 3;

		BYTE *idx = new BYTE[sz];
		for (int i = 0; i < sz; ) {
			idx[i] = ptr[i + 2] * 255;
			idx[i + 1] = ptr[i + 1] * 255;
			idx[i + 2] = ptr[i] * 255;

			i += 3;
		}

		if (!idx)
			return;

		int colorTablesize = 0;

		int biBitCount = 24;

		if (biBitCount == 8)
			colorTablesize = 1024;

		// The number of bytes in each line of image data to be stored is a multiple of 4.
		int lineByte = (_cx * biBitCount / 8 + 3) / 4 * 4;

		FILE *fp = fopen(fn, "wb");
		if (fp == 0)
			return;

		// Apply for bitmap file header structure variables and fill in the file header information.
		BITMAPFILEHEADER fileHead;
		fileHead.bfType = 0x4D42;// Type: BMP
		fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * _cy;
		fileHead.bfReserved1 = 0;
		fileHead.bfReserved2 = 0;

		// BfOffBits is the sum of the space required by the first three parts of an image file.
		fileHead.bfOffBits = 54 + colorTablesize;

		fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

		// Apply for bitmap header structure variables and fill in the header information.
		BITMAPINFOHEADER head;
		head.biBitCount = biBitCount;
		head.biClrImportant = 0;
		head.biClrUsed = 0;
		head.biCompression = 0;
		head.biHeight = _cy;
		head.biPlanes = 1;
		head.biSize = 40;
		head.biSizeImage = lineByte*_cy;
		head.biWidth = _cx;
		head.biXPelsPerMeter = 0;
		head.biYPelsPerMeter = 0;

		fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);
		fwrite(idx, _cy*lineByte, 1, fp);
		fclose(fp);
		delete[] idx;
	}

	void resetRGBnew() {
		if (_rgbsNew) delete[] _rgbsNew;
		_rgbsNew = new float[_cx*_cy * 3];

		// The target output color is initialized to red.
		for (int i = 0; i <_cx; i++) {
			for (int j = 0; j < _cy; j++) {
				float *p2 = _rgbsNew + (i*_cy + j) * 3;
				p2[0] = 1;
				p2[1] = 0;
				p2[2] = 0;
			}
		}
	}

	float* getRGBnew()
	{
		return _rgbsNew;
	}	

public:
	SceneData() {
		_cx = _cy = 0;
		_xyzs = _rgbs = NULL;
		_rgbsNew = NULL;
	}

	~SceneData() {
		if (_xyzs) delete[] _xyzs;
		if (_rgbs) delete[] _rgbs;
		if (_rgbsNew) delete[] _rgbsNew;
	}

	float *rgbs() { return _rgbs; }
	float *rgbsNew() { return _rgbsNew; }
	float *xyzs() { return _xyzs; }

	void saveNewRGB(char *fn) {
		saveAsBmp(_rgbsNew, fn);
	}
	
	void load(char *fname, int numofImg, bool updateHash=true) {
		char rgbFile[512], xyzFile[512];
	   sprintf(rgbFile, "%s%s", fname, ".rgb");
	   sprintf(xyzFile, "%s%s", fname, ".xyz");

	   if (updateHash)
		loadRGB(rgbFile);

	   loadXYZ(xyzFile);
	   
	   resetRGBnew();

	   float *p1 = _xyzs;
	   float *p2 = _rgbs;
	   int num = _cx*_cy;

	   for (int i = 0; i < num; i++) {
		   if (updateHash)
			gHashTab.push_back(xyz2rgb(vec3f(p1), vec3f(p2), numofImg, i));

		   _bound += vec3f(p1);
		   p1 += 3;
		   p2 += 3;
	   }
	}
};


class ProgScene : public SceneData {
public:
	float *_nearest;

protected:
	void resetNearest()
	{
		int sz = _cx*_cy;
		_nearest = new float[sz];

		for (int i = 0; i < sz; i++)
			_nearest[i] = 1000000;
	}

public:
	ProgScene() {
		_nearest = NULL;
	}

	~ProgScene() {
		if (_nearest)
			delete[] _nearest;
	}


	void load(char *fname) {
		SceneData::load(fname, -1, false);
		resetNearest();
	}

	void save(char *fname) {
		saveNewRGB(fname);
	}

	int getSize()
	{
		return _cx * _cy;
	}


	void update() {

#pragma omp parallel for schedule(dynamic, 5)
		for (int i = 0; i <_cx; i++) {
			printf("%d of %d done...\n", i, _cx);

			for (int j = 0; j < _cy; j++) {
				// Target: xyz.
				float *p1 = _xyzs + (i*_cy + j) * 3;
				// Target: new rgb.
				float *p2 = _rgbsNew + (i*_cy + j) * 3;
				// Target: the closest distance to each point.
				float *p3 = _nearest + (i*_cy + j);
				// Target: read color.
				float *p4 = _rgbs + (i*_cy + j) * 3;

				int idx, pidx;

				vec3f cr;
				bool ret = getColorCUDA(lTree, vec3f(p1), p3, cr, idx, pidx);
				if (ret) {
					p2[0] = cr.x;
					p2[1] = cr.y;
					p2[2] = cr.z;
				}
			}
		}

	}
};

SceneData scene[18];
ProgScene target;

// Copy lbvh from host to device.
__host__ t_linearBVH* copyBVHToDevice()
{
	// Copy lbvh to device.
	t_linearBVH* d_linear_bvh;
	t_linearBVH temp_bvh(*lTree);
	int size = lTree->num();
	t_linearBvhNode* rt;

	CHECK_ERROR(cudaMalloc((void**)&rt, sizeof(t_linearBvhNode)*size));
	CHECK_ERROR(cudaMemcpy(rt, lTree->_root, size * sizeof(t_linearBvhNode), cudaMemcpyHostToDevice));

	t_linearBvhNode* temp = temp_bvh._root;
	temp_bvh._root = rt;

	CHECK_ERROR(cudaMalloc((void**)&d_linear_bvh, sizeof(t_linearBVH)));
	CHECK_ERROR(cudaMemcpy(d_linear_bvh, &temp_bvh, sizeof(t_linearBVH), cudaMemcpyHostToDevice));
	temp_bvh._root = temp;

	return d_linear_bvh;

}
extern "C"
	void runCUDA(int width, int height, t_linearBVH* d_linear_bvh,
		float *_xyzs,
		float *_rgbsNew,
		float *_nearest);

int main()
{
	TIMING_BEGIN("Start loading...")
		// Load target xyz.
		target.load("all.bmp");
		// numofImg is 0.
		scene[0].load("0-55.bmp", 0);
	TIMING_END("Loading done...")
	
	TIMING_BEGIN("Start build_bvh...")
		build_bvh();
	TIMING_END("build_bvh done...")

	cudaEvent_t start, stop;
	CHECK_ERROR(cudaEventCreate(&start));
	CHECK_ERROR(cudaEventCreate(&stop));
	CHECK_ERROR(cudaEventRecord(start, 0));
	CHECK_ERROR(cudaEventSynchronize(start));

	// Copy lbvh to device.
	t_linearBVH* d_linear_bvh= copyBVHToDevice(); 	

	float *_xyzs;
	float *_rgbs;
	float *_rgbsNew;
	float *_nearest;
	CHECK_ERROR(cudaMalloc((void**)&_xyzs, sizeof(float) * 3 * target.getSize()));
	CHECK_ERROR(cudaMalloc((void**)&_rgbs, sizeof(float) * 3 * target.getSize()));
	CHECK_ERROR(cudaMalloc((void**)&_rgbsNew, sizeof(float) * 3 * target.getSize()));
	CHECK_ERROR(cudaMalloc((void**)&_nearest, sizeof(float) * target.getSize()));

	CHECK_ERROR(cudaMemcpy(_xyzs, target.xyzs(), sizeof(float) * 3 * target.getSize(), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(_rgbsNew, target.rgbsNew(), sizeof(float) * 3 * target.getSize(), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(_nearest, target._nearest, sizeof(float)  * target.getSize(), cudaMemcpyHostToDevice));

	// Timing
	CHECK_ERROR(cudaEventRecord(stop, 0));
	CHECK_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	std::cout << "cudaMalloc and cudaMemcpy:" << elapsedTime<<" <ms>" << std::endl;

	CHECK_ERROR(cudaEventRecord(start, 0));
	CHECK_ERROR(cudaEventSynchronize(start));
	runCUDA(target.width(),target.height(), d_linear_bvh, _xyzs,  _rgbsNew, _nearest);
	cudaDeviceSynchronize();
	CHECK_STATE("runCUDA kernel call");
	// Timing runtime kernel
	CHECK_ERROR(cudaEventRecord(stop, 0));
	CHECK_ERROR(cudaEventSynchronize(stop));
	CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	std::cout << "runCUDA:" << elapsedTime << " <ms>" << std::endl;

	CHECK_ERROR(cudaMemcpy(target.getRGBnew(), _rgbsNew, sizeof(float) * 3 * target.getSize(), cudaMemcpyDeviceToHost));

	//TIMING_BEGIN("Start rescanning...")
		//target.update();
	//TIMING_END("Rescaning done...")

	destroy_bvh();
	target.save("output.bmp");

	return 0;
}														 