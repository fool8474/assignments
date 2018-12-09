#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "imgproc.h"
#include "utils.h"
#include "examples.h"


int main() {
	cv::Mat input = cv::imread("samples/lena.bmp");
	cv::Mat face = cv::imread("samples/face.png");
	cv::Mat test = cv::imread("samples/lena.bmp");

	cv::Mat src_hsv, face_hsv;
	cvtColor(test, src_hsv, cv::COLOR_BGR2HSV);
	cvtColor(face, face_hsv, cv::COLOR_BGR2HSV);
	//cvtColor : �÷���ȯ �Լ� COLOR_BGR2HSV : BGR > HSV, COLOR_BGR2GRAY : BGR > Gray

	cv::Mat output;
	//input.copyTo(output);

	// Todo : imageproc.cpp�� �ִ� backprojectHistogram �Լ��� �ۼ��ϼ���
	IPCVL::IMG_PROC::backprojectHistogram(src_hsv, face_hsv, output);

	// display
	imshow("input", test);
	imshow("face", face);
	imshow("model output", output);
	cv::waitKey(0);

	return 0;
}

