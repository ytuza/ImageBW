#include <iostream>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_cuda.h"

using namespace std;
using namespace cv;

int main() {

	Mat input = imread("lena.jpg");

	cout << "Height: " << input.rows << ", Width: " << input.cols << ", Channels " << input.channels() << endl;

	namedWindow("image2", WINDOW_NORMAL);
	imshow("image2", input);


	conv_CUDA(input.data, input.rows, input.cols, input.channels());

	imwrite("conv_output.png", input);

	namedWindow("image", WINDOW_NORMAL);
	imshow("image", input);
	waitKey(0);

	//system("pause");


	return 0;

}