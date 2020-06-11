#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "image_cuda.h"

__global__ void image_bound(unsigned char* image, int Channels, int xm, int ym) {

	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * Channels;

	int sumR = 0, sumG = 0, sumB = 0;

	int idx0 = (x+1 + y * gridDim.x) * Channels;
	int idx1 = (x-1 + y * gridDim.x) * Channels;
	int idx2 = (x + (y+1) * gridDim.x) * Channels;
	int idx3 = (x + (y-1) * gridDim.x) * Channels;
	int idx4 = (x+1 + (y+1) * gridDim.x) * Channels;
	int idx5 = (x+1 + (y-1) * gridDim.x) * Channels;
	int idx6 = (x-1 + (y + 1) * gridDim.x) * Channels;
	int idx7 = (x-1 + (y - 1) * gridDim.x) * Channels;



	if (x == 0) {
		if (y == 0) {
			sumR = image[idx2] + image[idx4] + image[idx0];
			sumG = image[idx2+1] + image[idx4+1] + image[idx0+1];
			sumB = image[idx2+2] + image[idx4+2] + image[idx0+2];

			sumR = sumR / 3;
			sumG = sumG / 3;
			sumB = sumB / 3;
		}
		else if (y == ym-1) {
			sumR = image[idx3] + image[idx5] + image[idx0];
			sumG = image[idx3 + 1] + image[idx5 + 1] + image[idx0 + 1];
			sumB = image[idx3 + 2] + image[idx5 + 2] + image[idx0 + 2];
			sumR = sumR / 3;
			sumG = sumG / 3;
			sumB = sumB / 3;
		}
		else {
			sumR = image[idx2] + image[idx4] + image[idx0] + image[idx3]+ image[idx5];
			sumG = image[idx2 + 1] + image[idx4 + 1] + image[idx0 + 1] + image[idx3+1] + image[idx5+1];
			sumB = image[idx2 + 2] + image[idx4 + 2] + image[idx0 + 2] + image[idx3+2] + image[idx5+2];
			sumR = sumR / 5;
			sumG = sumG / 5;
			sumB = sumB / 5;
		}
	}else if (x == xm-1) {
		if (y == ym-1) {
			sumR = image[idx1] + image[idx7] + image[idx3];
			sumG = image[idx1 + 1] + image[idx7 + 1] + image[idx3 + 1];
			sumB = image[idx1 + 2] + image[idx7 + 2] + image[idx3 + 2];
			sumR = sumR / 3;
			sumG = sumG / 3;
			sumB = sumB / 3;
		}
		else if (y == 0) {
			sumR = image[idx1] + image[idx6] + image[idx2];
			sumG = image[idx1 + 1] + image[idx6 + 1] + image[idx2 + 1];
			sumB = image[idx1 + 2] + image[idx6 + 2] + image[idx2 + 2];
			sumR = sumR / 3;
			sumG = sumG / 3;
			sumB = sumB / 3;
		}
		else {
			sumR = image[idx2] + image[idx6] + image[idx1] + image[idx3] + image[idx7];
			sumG = image[idx2 + 1] + image[idx6 + 1] + image[idx1 + 1] + image[idx3 + 1] + image[idx7 + 1];
			sumB = image[idx2 + 2] + image[idx6 + 2] + image[idx1 + 2] + image[idx3 + 2] + image[idx7 + 2];
			sumR = sumR / 5;
			sumG = sumG / 5;
			sumB = sumB / 5;
		}
	}
	else if (y == ym - 1) {
		sumR = image[idx1] + image[idx7] + image[idx3] + image[idx5] + image[idx0];
		sumG = image[idx1 + 1] + image[idx7 + 1] + image[idx3 + 1] + image[idx5 + 1] + image[idx0 + 1];
		sumB = image[idx1 + 2] + image[idx7 + 2] + image[idx3 + 2] + image[idx5 + 2] + image[idx0 + 2];
		sumR = sumR / 5;
		sumG = sumG / 5;
		sumB = sumB / 5;
	}
	else if (y == 0) {
		sumR = image[idx1] + image[idx6] + image[idx2] + image[idx4] + image[idx0];
		sumG = image[idx1 + 1] + image[idx6+ 1] + image[idx2 + 1] + image[idx4 + 1] + image[idx0 + 1];
		sumB = image[idx1 + 2] + image[idx6 + 2] + image[idx2 + 2] + image[idx4 + 2] + image[idx0 + 2];
		sumR = sumR / 5;
		sumG = sumG / 5;
		sumB = sumB / 5;
	}
	else {
		sumR = image[idx1] + image[idx6] + image[idx2] + image[idx4] + image[idx0] + image[idx7] + image[idx3] + image[idx5];
		sumG = image[idx1 + 1] + image[idx6 + 1] + image[idx2 + 1] + image[idx4 + 1] + image[idx0 + 1] + image[idx7 + 1] + image[idx3 + 1] + image[idx5 + 1];
		sumB = image[idx1 + 2] + image[idx6 + 2] + image[idx2 + 2] + image[idx4 + 2] + image[idx0 + 2] + image[idx7 + 2] + image[idx3 + 2] + image[idx5 + 2];
		sumR = sumR / 8;
		sumG = sumG / 8;
		sumB = sumB / 8;
	}



	image[idx] = sumR;
	image[idx + 1] = sumG;
	image[idx + 2] = sumB;

}

__global__ void image_Inv(unsigned char* image, int Channels) {

	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		image[idx + i] = 255 - image[idx + i];
	}
	
}

__global__ void image_BW(unsigned char * image, int Channels) {

	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * Channels;

	int valgrey = (int)((0.21 * image[idx]) + (0.72 * image[idx + 1]) + (0.07 * image[idx + 2]));
	image[idx] = valgrey;
	image[idx+1] = valgrey;
	image[idx+2] = valgrey;

}


void conv_CUDA(unsigned char* input_image, int Height, int Width, int Channels) {
	unsigned char* pt_image = NULL;
	cudaMalloc((void**)&pt_image, Height*Width*Channels);

	cudaMemcpy(pt_image, input_image,  Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3 Grid(Width, Height);

	//image_BW <<<  Grid, 1 >>> (pt_image, Channels);
	image_bound << <  Grid, 1 >> > (pt_image, Channels, Width, Height);

	cudaMemcpy(input_image, pt_image, Height * Width * Channels, cudaMemcpyDeviceToHost);
	
	cudaFree(pt_image);
}

