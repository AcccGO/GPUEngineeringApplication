#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bvh.h"

__device__ __host__ bool findNearest2CUDA(t_linearBVH* d_linear_bvh, xyz2rgb &org, xyz2rgb &ret)
{
	return d_linear_bvh->query(org, ret);
}

extern "C"
__device__ __host__ bool getColorCUDA(t_linearBVH* d_linear_bvh, vec3f &xyz, float *nearest, vec3f &cr, int &idx, int &pidx)
{
	xyz2rgb input(xyz, vec3f(), idx, pidx), ret;

	bool find = findNearest2CUDA(d_linear_bvh,input, ret);

	if (!find)
		return false;

	vec3f now = ret.xyz();
	float dist = vdistance(xyz, now);
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

__global__ void updateTargetColors(int width, int height, t_linearBVH* d_linear_bvh,
	float *_xyzs,
	float *_rgbsNew,
	float *_nearest)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int offset = iy * width + ix;
	//printf("ix is %d, iy is %d, offset is %d, total is %d\n", ix, iy, offset, width*height);
	if (offset >= width*height)
		return;

	float *p1 = _xyzs + offset * 3;
	float *p2 = _rgbsNew + offset * 3;
	float *p3 = _nearest + offset;

	int idx, pidx;
	// Output color.
	vec3f cr;
	bool ret = getColorCUDA(d_linear_bvh,vec3f(p1), p3, cr, idx, pidx);
	if (ret) {
		//printf("CUDA:%d of %d done success\n", offset, width*height);
		p2[0] = cr.x;
		p2[1] = cr.y;
		p2[2] = cr.z;
	}
	//else
	//{
	//	printf("CUDA:%d of %d done fail\n", offset, width*height);
	//}
}

extern "C"
void runCUDA(int width, int height, t_linearBVH* d_linear_bvh,
	float *_xyzs,
	float *_rgbsNew,
	float *_nearest)
{
	// Init block
	int dimx = 16;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	updateTargetColors <<<grid, block >>> (width,height,d_linear_bvh,_xyzs,_rgbsNew,_nearest);
}

