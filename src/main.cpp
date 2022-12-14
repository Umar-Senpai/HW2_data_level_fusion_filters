#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include <numeric>
#include <math.h>

// https://www.researchgate.net/publication/3324796_Weighted_median_filters_A_tutorial

// TAKEN FROM https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html 
# define M_PI           3.14159265358979323846
cv::Scalar getMSSIM( const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = cv::mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}

float Median(std::vector<float> &v)
{
    std::size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

void calculateAndPrintErrors(cv::Mat& output, cv::Mat& gt_image){
	long int SSD = 0;
	double MSE = 0;
	double RMSE = 0;
	double PSNR = 0;
	cv::Scalar SSIM;
	cv::Mat NCC;
	NCC.create( 1, 1, CV_32FC1 );

	const auto width = output.cols;
	const auto height = output.rows;

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			int v = output.at<uchar>(r, c) - gt_image.at<uchar>(r, c);
			SSD += (v * v);
		}
	}

	MSE = SSD / (width * height);
	RMSE = std::sqrt(MSE);
	int max_pixel_value = 255;
	PSNR = (10 * std::log10((max_pixel_value * max_pixel_value) / MSE));
	SSIM = getMSSIM(output, gt_image);
	cv::matchTemplate(output, gt_image, NCC, cv::TM_CCORR_NORMED);

	std::cout << "SSD: " << SSD << std::endl;
	std::cout << "MSE: " << MSE << std::endl;
	std::cout << "RMSE: " << RMSE << std::endl;
	std::cout << "PSNR: " << PSNR << std::endl;
	std::cout << "SSIM: " << SSIM[0] << std::endl;
	std::cout << "NCC: " << NCC.at<float>(0) << std::endl;
}

void calculateAndWriteErrors(cv::Mat& output, cv::Mat& gt_image, std::ofstream& out){
	long int SSD = 0;
	double MSE = 0;
	double RMSE = 0;
	double PSNR = 0;
	cv::Scalar SSIM;

	const auto width = output.cols;
	const auto height = output.rows;

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			int v = output.at<uchar>(r, c) - gt_image.at<uchar>(r, c);
			SSD += (v * v);
		}
	}

	MSE = SSD / (width * height);
	RMSE = std::sqrt(MSE);
	int max_pixel_value = 255;
	PSNR = (10 * std::log10((max_pixel_value * max_pixel_value) / MSE));
	SSIM = getMSSIM(output, gt_image);

	out << "RMSE: " << RMSE << std::endl;
	out << "PSNR: " << PSNR << std::endl;
	out << "SSIM: " << SSIM[0] << std::endl;
}

void createGaussianMask(const int& window_size, cv::Mat& mask, int& sum_mask){
	cv::Size mask_size(window_size, window_size);
	mask = cv::Mat(mask_size, CV_8UC1);
	const double sigma = window_size / 2.5;

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			int x = r - window_size / 2;
			int y = c - window_size / 2;
			mask.at<uchar>(r, c) = 255 * std::exp(-(x * x + y * y) / (2 * sigma * sigma)); // Box filter
		}
	}

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			sum_mask += static_cast<int>(mask.at<uchar>(r, c));
		}
	}
}

void OurFiler(const cv::Mat& input, cv::Mat& output) {

	const auto width = input.cols;
	const auto height = input.rows;

	const int window_size = 7;

	// TEMPORARY CODE
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0; // normalize filtering

	createGaussianMask(window_size, mask, sum_mask);

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {

			// box filter
			int sum = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					int weight = static_cast<int>(mask.at<uchar>(i + window_size / 2, j + window_size / 2));
					sum += intensity * weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_mask;

		}
	}
}

void OurBilateralFiler(const cv::Mat& input, cv::Mat& output) {

	const auto width = input.cols;
	const auto height = input.rows;

	const int window_size = 7;

	// TEMPORARY CODE
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0; // normalize filtering

	createGaussianMask(window_size, mask, sum_mask);

	const float sigmaRange = 50; // TODO: Experiment
	const float sigmaRangeSq = sigmaRange * sigmaRange;
	float range_mask[256];
	// compute range kernel
	for (int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			// box filter
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					// TODO: compute range diff to center pixel value
					int diff = std::abs(intensity_center - intensity); // 0..255
					// TODO: compute the range kernel's value
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2, 
						j + window_size / 2));

					// ... combine weights ...
					float weight = weight_range * weight_spatial;
					sum += intensity * weight;
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask;

		}
	}
}

void OurBilateralMedianFiler(const cv::Mat& input, cv::Mat& output) {

	const auto width = input.cols;
	const auto height = input.rows;

	const int window_size = 7;

	// TEMPORARY CODE
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0; // normalize filtering

	createGaussianMask(window_size, mask, sum_mask);

	const float sigmaRange = 50; // TODO: Experiment
	const float sigmaRangeSq = sigmaRange * sigmaRange;
	float range_mask[256];
	// compute range kernel
	for (int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			// box filter
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			std::vector<float> kernel_histogram(window_size * window_size);
			std::vector<float> weight_histogram(window_size * window_size);
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					// TODO: compute range diff to center pixel value
					int diff = std::abs(intensity_center - intensity); // 0..255
					// TODO: compute the range kernel's value
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2, 
						j + window_size / 2));

					// ... combine weights ...
					float weight = weight_range * weight_spatial;
					kernel_histogram[(i + window_size / 2 ) * window_size + j + window_size / 2] = intensity;
					weight_histogram[(i + window_size / 2 ) * window_size + j + window_size / 2] = weight;
					sum_Bilateral_mask += weight;
				}
			}
			float thres = sum_Bilateral_mask / 2;
			float sum1 = 0;
			int index = 0;

			std::vector<int> indices(weight_histogram.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::sort(indices.begin(), indices.end(),
					[&](int A, int B) -> bool {
							return kernel_histogram[A] > kernel_histogram[B];
						});
						
			while(sum1 < thres) {
				sum1 += weight_histogram[indices[index]];
				index++;
			}
			// std::cout << Median(kernel_histogram) << std::endl;
			output.at<uchar>(r, c) = kernel_histogram[indices[index]];

		}
	}
}

void OurJointBilateralFiler(const cv::Mat& input, const cv::Mat& depth, cv::Mat& output, const int window_size = 7, const float sigmaRange = 50) {

	const auto width = input.cols;
	const auto height = input.rows;

	// const int window_size = 7;

	cv::Mat mask;
	int sum_mask = 0; // normalize filtering

	createGaussianMask(window_size, mask, sum_mask);

	// const float sigmaRange = 50; // TODO: Experiment
	const float sigmaRangeSq = sigmaRange * sigmaRange;
	float range_mask[256];
	// compute range kernel
	for (int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			// box filter
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					// TODO: compute range diff to center pixel value
					int diff = std::abs(intensity_center - intensity); // 0..255
					// TODO: compute the range kernel's value
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2, 
						j + window_size / 2));

					// ... combine weights ...
					float weight = weight_range * weight_spatial;
					sum += static_cast<int>(depth.at<uchar>(r + i, c + j)) * weight;
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask;

		}
	}
}

void OurJointBilateralMedianFiler(const cv::Mat& input, const cv::Mat& depth, cv::Mat& output, const int window_size = 7, const float sigmaRange = 50) {

	const auto width = input.cols;
	const auto height = input.rows;

	// const int window_size = 7;

	// TEMPORARY CODE
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0; // normalize filtering

	createGaussianMask(window_size, mask, sum_mask);

	// const float sigmaRange = 50; // TODO: Experiment
	const float sigmaRangeSq = sigmaRange * sigmaRange;
	float range_mask[256];
	// compute range kernel
	for (int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			// box filter
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			std::vector<float> kernel_histogram(window_size * window_size);
			std::vector<float> weight_histogram(window_size * window_size);
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					// TODO: compute range diff to center pixel value
					int diff = std::abs(intensity_center - intensity); // 0..255
					// TODO: compute the range kernel's value
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2, 
						j + window_size / 2));

					// ... combine weights ...
					float weight = weight_range * weight_spatial;
					kernel_histogram[(i + window_size / 2 ) * window_size + j + window_size / 2] = static_cast<int>(depth.at<uchar>(r + i, c + j));
					weight_histogram[(i + window_size / 2 ) * window_size + j + window_size / 2] = weight;
					sum_Bilateral_mask += weight;
				}
			}
			float thres = sum_Bilateral_mask / 2;
			float sum1 = 0;
			int index = 0;

			std::vector<int> indices(weight_histogram.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::sort(indices.begin(), indices.end(),
					[&](int A, int B) -> bool {
							return kernel_histogram[A] > kernel_histogram[B];
						});
						
			while(sum1 < thres) {
				sum1 += weight_histogram[indices[index]];
				index++;
			}
			// std::cout << Median(kernel_histogram) << std::endl;
			output.at<uchar>(r, c) = kernel_histogram[indices[index]];

		}
	}
}

void OurJointBilateralUpsamplingFiler(const cv::Mat& input, const cv::Mat& depth, cv::Mat& output, const int window_size = 7, const float sigmaRange = 50) {

	const auto width = input.cols;
	const auto height = input.rows;

	const auto lowres_height = depth.rows;

	const int upsampling_factor = height / lowres_height;

	// const int window_size = 7;

	// TEMPORARY CODE
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0; // normalize filtering

	createGaussianMask(window_size, mask, sum_mask);

	// const float sigmaRange = 50; // TODO: Experiment
	const float sigmaRangeSq = sigmaRange * sigmaRange;
	float range_mask[256];
	// compute range kernel
	for (int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			// box filter
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					// TODO: compute range diff to center pixel value
					int diff = std::abs(intensity_center - intensity); // 0..255
					// TODO: compute the range kernel's value
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2, 
						j + window_size / 2));

					// ... combine weights ...
					float weight = weight_range * weight_spatial;
					sum += static_cast<int>(depth.at<uchar>((r + i)/upsampling_factor, (c + j)/upsampling_factor)) * weight;
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask;

		}
	}
}

void OurJointBilateralMedianUpsamplingFiler(const cv::Mat& input, const cv::Mat& depth, cv::Mat& output, const int window_size = 7, const float sigmaRange = 50) {

	const auto width = input.cols;
	const auto height = input.rows;

	const auto lowres_height = depth.rows;

	const int upsampling_factor = height / lowres_height;

	// const int window_size = 7;

	// TEMPORARY CODE
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0; // normalize filtering

	createGaussianMask(window_size, mask, sum_mask);

	// const float sigmaRange = 50; // TODO: Experiment
	const float sigmaRangeSq = sigmaRange * sigmaRange;
	float range_mask[256];
	// compute range kernel
	for (int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			// box filter
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			std::vector<float> kernel_histogram(window_size * window_size);
			std::vector<float> weight_histogram(window_size * window_size);
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					// TODO: compute range diff to center pixel value
					int diff = std::abs(intensity_center - intensity); // 0..255
					// TODO: compute the range kernel's value
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2, 
						j + window_size / 2));

					// ... combine weights ...
					float weight = weight_range * weight_spatial;
					kernel_histogram[(i + window_size / 2 ) * window_size + j + window_size / 2] = static_cast<int>(depth.at<uchar>((r + i)/upsampling_factor, (c + j)/upsampling_factor));
					weight_histogram[(i + window_size / 2 ) * window_size + j + window_size / 2] = weight;
					sum_Bilateral_mask += weight;
				}
			}
			float thres = sum_Bilateral_mask / 2;
			float sum1 = 0;
			int index = 0;

			std::vector<int> indices(weight_histogram.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::sort(indices.begin(), indices.end(),
					[&](int A, int B) -> bool {
							return kernel_histogram[A] > kernel_histogram[B];
						});
						
			while(sum1 < thres) {
				sum1 += weight_histogram[indices[index]];
				index++;
			}
			// std::cout << Median(kernel_histogram) << std::endl;
			output.at<uchar>(r, c) = kernel_histogram[indices[index]];

		}
	}
}

void Iterative_Upsampling(const cv::Mat& input, const cv::Mat& depth, cv::Mat& output, const int window_size = 7, const float sigmaRange = 50) {
	// applying the joint bilateral filter to upsample a depth image, guided by an RGB image -- iterative upsampling
	int uf = log2(input.rows / depth.rows); // upsample factor
	cv::Mat D = depth.clone(); // lowres depth image
	cv::Mat I = input.clone(); // highres rgb image
	for (int i = 0; i < uf; ++i)
	{
		cv::resize(D, D, D.size() * 2); // doubling the size of the depth image
		cv::resize(I, I, D.size());		// resizing the rgb image to depth image size
		OurJointBilateralFiler(I, D, D, window_size, sigmaRange); // applying the joint bilateral filter with changed size depth and rbg images
	}
	cv::resize(D, D, input.size()); // in the end resizing the depth image to rgb image size
	OurJointBilateralFiler(input, D, output, window_size, sigmaRange); // applying the joint bilateral filter with full res. size images
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << "PATH_TO_IMAGE_PAIR OUTPUT_FILE" << std::endl;
		return 1;
	}

	std::string dataFolderPath = argv[1];
	std::string output_file = argv[2];
	
	// TUNABLE PARAMETERS
	int window_size = 7;
	float sigmaRange = 50;

	cv::Mat im = cv::imread("../data/lena.png", cv::IMREAD_GRAYSCALE);

	cv::Mat color = cv::imread(dataFolderPath + "view1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat depth = cv::imread(dataFolderPath + "disp1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat lowres_depth = cv::imread(dataFolderPath + "lowres_disp1.png", cv::IMREAD_GRAYSCALE);

	if (im.data == nullptr) {
		std::cerr << "Failed to load image" << std::endl;
	}

	cv::Mat gt_image = im.clone();
	cv::Mat gt_depth = depth.clone();

	//cv::imshow("im", im);
	//cv::waitKey();

	cv::Mat noise(im.size(), im.type());
	uchar mean = 0;
	uchar stddev = 25;
	cv::randn(noise, mean, stddev);


	im += noise;

	cv::imshow("Image with Noise", im);
	//cv::waitKey();

	cv::Mat output = cv::Mat::zeros(im.rows, im.cols, CV_8UC1);
	// gaussian
	// cv::GaussianBlur(im, output, cv::Size(7, 7), 0, 0);
	// cv::imshow("gaussian", output);
	//cv::waitKey();

	// // median
	// cv::medianBlur(im, output, 3);
	// cv::imshow("median", output);
	// //cv::waitKey();

	// // bilateral
	// double window_size = 11;
	// cv::bilateralFilter(im, output, window_size, 2 * window_size, window_size / 2);
	// cv::imshow("bilateral", output);

	OurFiler(im, output);
	cv::imshow("OurFiler", output);

	std::cout << "----- OUR GAUSSIAN FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_image);

	OurBilateralFiler(im, output);
	cv::imshow("OurBilateralFiler", output);

	std::cout << "----- OUR BILATERAL FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_image);

	OurBilateralMedianFiler(im, output);
	cv::imshow("OurBilateralMedianFiler", output);

	std::cout << "----- OUR BILATERAL MEDIAN FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_image);

	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	double matching_time;
	std::stringstream outTime;
	outTime << output_file << "_processing_time.txt";
	std::ofstream outfileTime(outTime.str());
	output = cv::Mat::zeros(color.rows, color.cols, CV_8UC1);

	matching_time = (double)cv::getTickCount();
	OurJointBilateralFiler(color, depth, output, window_size, sigmaRange);
	matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  	outfileTime << "JB: " << matching_time << " seconds" << std::endl;
	cv::imshow("OurJointBilateralFiler", output);

	std::cout << "----- OUR JOINT BILATERAL FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_depth);

	matching_time = (double)cv::getTickCount();
	OurJointBilateralMedianFiler(color, depth, output, window_size, sigmaRange);
	matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  	outfileTime << "JBM: " << matching_time << " seconds" << std::endl;
	cv::imshow("OurJointBilateralMedianFiler", output);

	std::cout << "----- OUR JOINT BILATERAL MEDIAN FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_depth);

	matching_time = (double)cv::getTickCount();
	OurJointBilateralUpsamplingFiler(color, lowres_depth, output, window_size, sigmaRange);
	matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  	outfileTime << "JBU: " << matching_time << " seconds" << std::endl;
	cv::imshow("OurJointBilateralUpsamplingFiler", output);

	std::cout << "----- OUR JOINT BILATERAL UPSAMPLING FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_depth);

	matching_time = (double)cv::getTickCount();
	OurJointBilateralMedianUpsamplingFiler(color, lowres_depth, output, window_size, sigmaRange);
	matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  	outfileTime << "JBMU: " << matching_time << " seconds" << std::endl;
	cv::imshow("OurJointBilateralMedianUpsamplingFiler", output);

	std::cout << "----- OUR JOINT BILATERAL MEDIAN UPSAMPLING FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_depth);

	matching_time = (double)cv::getTickCount();
	Iterative_Upsampling(color, lowres_depth, output, window_size, sigmaRange);
	matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  	outfileTime << "IU: " << matching_time << " seconds" << std::endl;
	cv::imshow("Iterative_Upsampling", output);
	std::cout << "----- Iterative_Upsampling FILTER ERRORS -----" << std::endl;
	calculateAndPrintErrors(output, gt_depth);

	cv::waitKey();
	cv::destroyAllWindows();

	// Evaluate 12 pairs with different window sizes
	std::string dataset[12]
        = { "Aloe", "Art", "Baby1", "Books", "Bowling1", "Dolls", "Flowerpots", "Laundry", "Moebius", "Monopoly", "Reindeer", "Wood1" };

	
	// double window_size_arr[3] = {3, 5, 7};
	// // double window_size_arr[3] = {3, 5, 7};
	// for (int i = 0; i < 12; ++i){
	// 	std::cout
	// 		<< "Upsampling on the given dataset "
	// 		<< std::ceil(((i + 1) / 12.0 * 100)) << "%\r"
	// 		<< std::flush;

	// 	dataFolderPath = "../data/" + dataset[i];
	// 	std::string output_folder = "../results/" + dataset[i];
	// 	cv::Mat image = cv::imread(dataFolderPath + "/view1.png", cv::IMREAD_GRAYSCALE);
	// 	cv::Mat disparity = cv::imread(dataFolderPath + "/disp1.png", cv::IMREAD_GRAYSCALE);
	// 	cv::Mat lowres_disparity = cv::imread(dataFolderPath + "/lowres_disp1.png", cv::IMREAD_GRAYSCALE);
		
	// 	for (int j = 0; j < 3; ++j){	
	// 		window_size = window_size_arr[j];
	// 		output = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	// 		double time_taken = 0;

	// 		std::string output_file = output_folder + "_window_" + std::to_string(window_size);

	// 		std::stringstream errors;
	// 		errors << output_file << "_errors.txt";
	// 		std::ofstream outfileErrors(errors.str());

	// 		std::stringstream time;
	// 		time << output_file << "_processing_time.txt";
	// 		std::ofstream outfileTime(time.str());

	// 		time_taken = (double)cv::getTickCount();
	// 		OurJointBilateralUpsamplingFiler(image, lowres_disparity, output, window_size, sigmaRange);
	// 		time_taken = ((double)cv::getTickCount() - time_taken)/cv::getTickFrequency();
	// 		outfileTime << "JBU: " << time_taken << " seconds" << std::endl;
	// 		calculateAndWriteErrors(output, disparity, outfileErrors);
	// 		cv::imwrite(output_file + "_JBU.png", output);

	// 		time_taken = (double)cv::getTickCount();
	// 		OurJointBilateralMedianUpsamplingFiler(image, lowres_disparity, output, window_size, sigmaRange);
	// 		time_taken = ((double)cv::getTickCount() - time_taken)/cv::getTickFrequency();
	// 		outfileTime << "JBMU: " << time_taken << " seconds" << std::endl;
	// 		calculateAndWriteErrors(output, disparity, outfileErrors);
	// 		cv::imwrite(output_file + "_JBMU.png", output);

	// 		time_taken = (double)cv::getTickCount();
	// 		Iterative_Upsampling(image, lowres_disparity, output, window_size, sigmaRange);
	// 		time_taken = ((double)cv::getTickCount() - time_taken)/cv::getTickFrequency();
	// 		outfileTime << "IU: " << time_taken << " seconds" << std::endl;
	// 		calculateAndWriteErrors(output, disparity, outfileErrors);
	// 		cv::imwrite(output_file + "_IU.png", output);
	// 	}

	// }

	double sigma_arr[10] = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0};
	for (int i = 0; i < 3; ++i){
		std::cout
			<< "Upsampling on the given dataset "
			<< std::ceil(((i + 1) / 3.0 * 100)) << "%\r"
			<< std::flush;

		dataFolderPath = "../data/" + dataset[i];
		std::string output_folder = "../results/sigma/" + dataset[i];
		cv::Mat image = cv::imread(dataFolderPath + "/view1.png", cv::IMREAD_GRAYSCALE);
		cv::Mat disparity = cv::imread(dataFolderPath + "/disp1.png", cv::IMREAD_GRAYSCALE);
		cv::Mat lowres_disparity = cv::imread(dataFolderPath + "/lowres_disp1.png", cv::IMREAD_GRAYSCALE);
		
		for (int j = 0; j < 10; ++j){	
			window_size = 7;
			float sigma_size = sigma_arr[j];
			output = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
			double time_taken = 0;

			std::string output_file = output_folder + "_window_" + std::to_string(window_size) + "_sigma_" + std::to_string(sigma_size);

			std::stringstream errors;
			errors << output_file << "_errors.txt";
			std::ofstream outfileErrors(errors.str());

			std::stringstream time;
			time << output_file << "_processing_time.txt";
			std::ofstream outfileTime(time.str());

			time_taken = (double)cv::getTickCount();
			Iterative_Upsampling(image, lowres_disparity, output, window_size, sigma_size);
			time_taken = ((double)cv::getTickCount() - time_taken)/cv::getTickFrequency();
			outfileTime << "IU: " << time_taken << " seconds" << std::endl;
			calculateAndWriteErrors(output, disparity, outfileErrors);
			cv::imwrite(output_file + "_IU.png", output);
		}

	}

	return 0;
}