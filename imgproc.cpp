#include "imgproc.h"

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, int* histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					histogram[inputMat.at<uchar>(y, x)]++;
					//�� Mat�� Loop�� ���Ͽ� ����, Histogram ����
				}
			}
		}
		void backprojectHistogram(cv::InputArray src_hsv, cv::InputArray face_hsv, cv::OutputArray dst) {
			cv::Mat srcMat = src_hsv.getMat();
			cv::Mat faceMat = face_hsv.getMat();
			dst.create(srcMat.size(), CV_64FC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.)); // �迭�� ��ü�� 0���� ����

			std::vector<cv::Mat> channels;
			split(src_hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			double model_hist[64][64] = { { 0., } };
			double input_hist[64][64] = { { 0., } };

			// hs 2���� ������׷��� ����ϴ� �Լ�.
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);


			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : ����ȭ�� h,s ���� ��� histogram�� ���� ���մϴ�. 

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
			split(hsv, channels); //channels[0] ����(H), 1 ä��(S), 2 ��(V)
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2���� ������׷��� �׽��ϴ�. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : ����ȭ�� h,s ���� ��� histogram�� ���� ���մϴ�. 
					histogram[UTIL::quantize(mat_h.at<uchar>(y, x))][UTIL::quantize(mat_s.at<uchar>(y, x))]++;

					// hint 1 : ����ȭ �� UTIL::quantize() �Լ��� �̿��ؼ� mat_h, mat_s�� ���� ����ȭ��ŵ�ϴ�. 
				}
			}

			// ������׷��� (hsv.rows * hsv.cols)���� ����ȭ�մϴ�. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					histogram[j][i] = histogram[j][i] / (hsv.rows * hsv.cols); // ��� ���ϸ� 1
				}
			}
		}
	}  // namespace IMG_PROC
}

