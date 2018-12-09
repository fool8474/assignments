#include "imgproc.h"

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, int* histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					histogram[inputMat.at<uchar>(y, x)]++;
					//각 Mat에 Loop를 통하여 접근, Histogram 증가
				}
			}
		}
		void backprojectHistogram(cv::InputArray src_hsv, cv::InputArray face_hsv, cv::OutputArray dst) {
			cv::Mat srcMat = src_hsv.getMat();
			cv::Mat faceMat = face_hsv.getMat();
			dst.create(srcMat.size(), CV_64FC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.)); // 배열의 전체를 0으로 변경

			std::vector<cv::Mat> channels;
			split(src_hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			double model_hist[64][64] = { { 0., } };
			double input_hist[64][64] = { { 0., } };

			// hs 2차원 히스토그램을 계산하는 함수.
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);


			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 

					/** your code here! **/
					int h = UTIL::quantize(mat_h.at<uchar>(y, x));
					int s = UTIL::quantize(mat_s.at<uchar>(y, x));

					outputProb.at<double>(y, x) = UTIL::h_r(model_hist, input_hist, h, s);
				}
			}
		}

		void calcHist_hs(cv::InputArray src_hsv, double histogram[][64]) {
			cv::Mat hsv = src_hsv.getMat();
			std::vector<cv::Mat> channels;
			split(hsv, channels); //channels[0] 색상(H), 1 채도(S), 2 명도(V)
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2차원 히스토그램을 쌓습니다. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					histogram[UTIL::quantize(mat_h.at<uchar>(y, x))][UTIL::quantize(mat_s.at<uchar>(y, x))]++;

					// hint 1 : 양자화 시 UTIL::quantize() 함수를 이용해서 mat_h, mat_s의 값을 양자화시킵니다. 
				}
			}

			// 히스토그램을 (hsv.rows * hsv.cols)으로 정규화합니다. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					histogram[j][i] = histogram[j][i] / (hsv.rows * hsv.cols); // 모두 더하면 1
				}
			}
		}
	}  // namespace IMG_PROC
}

