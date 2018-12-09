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

		void thresh_binary(cv::InputArray src, cv::OutputArray dst, const int& threshold) {
			cv::Mat srcMat = src.getMat();
			dst.create(srcMat.size(), CV_8UC1);
			cv::Mat outputMat = dst.getMat();

			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {

					if (srcMat.at<uchar>(y, x) < threshold) {
						outputMat.at<uchar>(y, x) = 0;
					}
					else {
						outputMat.at<uchar>(y, x) = 255;
					}
				}
			}
		}

		void thresh_otsu(cv::InputArray src, cv::OutputArray dst) {

			cv::Mat srcMat = src.getMat();
			dst.create(srcMat.size(), CV_8UC1);
			cv::Mat outputMat = dst.getMat();
			outputMat.setTo(cv::Scalar(0.));

			double histogramNormalized[256] = { 0, };
			UTIL::calcNormedHist(src, histogramNormalized);

			double w[256]       = { 0, }, 
				  u0[256]       = { 0, }, 
				  u1[256]       = { 0, }, 
				  vBetween[256] = { 0, };

			double    u = 0.0, 
				 curMax = 0.0;
			int threshold = 0;

			for (int t = 0; t < 256; t++) {
				u += (t*histogramNormalized[t]);
			}

			w[0] = histogramNormalized[0];
			u0[0] = 0.0;

			for (int t = 1; t < 256; t++) {

				w[t] = w[t - 1] + histogramNormalized[t];

				if (w[t] == 0 || (1 - w[t]) == 0)
					continue;

				u0[t] = (w[t - 1] * u0[t - 1] + t * histogramNormalized[t]) / w[t];
				u1[t] = (u - w[t] * u0[t]) / (1 - w[t]);
				vBetween[t] = w[t] * (1 - w[t]) * (pow(u0[t] - u1[t], 2));
				
				if (vBetween[t] > curMax) {
					curMax = vBetween[t];
					threshold = t;
				}
			}

			thresh_binary(srcMat, outputMat, threshold);
		}

		void flood_fill4(cv::Mat& l, const int& j, const int& i, const int& label) {

		}
		void flood_fill8(cv::Mat& l, const int& j, const int& i, const int& label) {

		}
		void efficient_flood_fill4(cv::Mat& l, const int& j, const int& i, const int& label) {

		}
		void flood_fill(cv::InputArray src, cv::OutputArray dst, const UTIL::CONNECTIVITIES& direction) {

		}

	}  // namespace IMG_PROC


}

