#include "pclviewer.h"
#include "ui_pclviewer.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include <numeric>
#include <math.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <thread>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/obj_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>

#if VTK_MAJOR_VERSION > 8
#include <vtkGenericOpenGLRenderWindow.h>
#endif

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

void OurJointBilateralFiler(const cv::Mat& input, const cv::Mat& depth, cv::Mat& output, const int window_size = 7, const float sigmaRange = 50) {

	const auto width = input.cols;
	const auto height = input.rows;

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

void Iterative_Upsampling(const cv::Mat& input, const cv::Mat& depth, cv::Mat& output, const int window_size=7, const float sigma=50) {
	// applying the joint bilateral filter to upsample a depth image, guided by an RGB image -- iterative upsampling
	int uf = log2(input.rows / depth.rows); // upsample factor
	cv::Mat D = depth.clone(); // lowres depth image
	cv::Mat I = input.clone(); // highres rgb image
	for (int i = 0; i < uf; ++i)
	{
		cv::resize(D, D, D.size() * 2); // doubling the size of the depth image
		cv::resize(I, I, D.size());		// resizing the rgb image to depth image size
		OurJointBilateralFiler(I, D, D, window_size, sigma); // applying the joint bilateral filter with changed size depth and rbg images
	}
	cv::resize(D, D, input.size()); // in the end resizing the depth image to rgb image size
	OurJointBilateralFiler(input, D, output, window_size, sigma); // applying the joint bilateral filter with full res. size images
}

pcl::PointCloud<pcl::PointXYZ> Disparity2PointCloud_PCL(
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;

  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      if (disparities.at<uchar>(i, j) == 0) continue;

      const double d = disparities.at<uchar>(i, j) + dmin;
      const double u1 = j - width/2; // Coordinates of horizontal axis
      const double u2 = u1 - d;
      const double v1 = i - height/2; // Coordinates of vertical axis

      const double Z = (baseline * focal_length) / d;
      const double X = -(baseline * (u1 + u2)) / (2 * d);
      const double Y = (baseline * v1) / d;

      cloud.push_back(pcl::PointXYZ(X, Y, Z));
    }
  }
  return cloud;
}

double calculateRMSE(cv::Mat& output, cv::Mat& gt_image){
	long int SSD = 0;
	double MSE = 0;
	double RMSE = 0;

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
	return RMSE;
}

double calculatePSNR(cv::Mat& output, cv::Mat& gt_image){
	long int SSD = 0;
	double MSE = 0;
  double PSNR = 0;

	const auto width = output.cols;
	const auto height = output.rows;

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			int v = output.at<uchar>(r, c) - gt_image.at<uchar>(r, c);
			SSD += (v * v);
		}
	}

	MSE = SSD / (width * height);
  int max_pixel_value = 255;
	PSNR = (10 * std::log10((max_pixel_value * max_pixel_value) / MSE));
	return PSNR;
}

PCLViewer::PCLViewer (QWidget *parent) :
  QMainWindow (parent),
  ui (new Ui::PCLViewer)
{
  cv::Mat color = cv::imread("../data/Art/view1.png", 0);
	cv::Mat depth_low_res = cv::imread("../data/Art/lowres_disp1.png", 0);
  cv::Mat gt_disparity = cv::imread("../data/Art/disp1.png", 0);

  cv::Mat output = cv::Mat::zeros(color.rows, color.cols, CV_8UC1);
  const int window_size = 7;
  const float sigma = 50;
  new_window_size = window_size;
  new_sigma = sigma;
	Iterative_Upsampling(color, depth_low_res, output, window_size, sigma);

  const double focal_length = 3740;
  const double baseline = 0.160;

  // stereo estimation parameters
  const int dmin = 200;

  int height = output.size().height;
  int width = output.size().width;

  ui->setupUi (this);
  this->setWindowTitle ("PCL viewer");

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  *cloud = Disparity2PointCloud_PCL(
    height, width, output,
    window_size, dmin, baseline, focal_length);

  // Set up the QVTK window  
#if VTK_MAJOR_VERSION > 8
  auto renderer = vtkSmartPointer<vtkRenderer>::New();
  auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  viewer.reset(new pcl::visualization::PCLVisualizer(renderer, renderWindow, "viewer", false));
  ui->qvtkWidget->setRenderWindow(viewer->getRenderWindow());
  viewer->setupInteractor(ui->qvtkWidget->interactor(), ui->qvtkWidget->renderWindow());
#else
  viewer.reset(new pcl::visualization::PCLVisualizer("viewer", false));
  ui->qvtkWidget->SetRenderWindow(viewer->getRenderWindow());
  viewer->setupInteractor(ui->qvtkWidget->GetInteractor(), ui->qvtkWidget->GetRenderWindow());
#endif

  // Connect sliders and their functions
  connect (ui->horizontalSlider_sigma, SIGNAL (valueChanged (int)), this, SLOT (sigmaSliderValueChanged (int)));
  connect (ui->horizontalSlider_window, SIGNAL (valueChanged (int)), this, SLOT (windowSliderValueChanged (int)));
  connect (ui->horizontalSlider_sigma, SIGNAL (sliderReleased ()), this, SLOT (onSliderReleased ()));
  connect (ui->horizontalSlider_window, SIGNAL (sliderReleased ()), this, SLOT (onSliderReleased ()));

  // Connect point size slider
  connect (ui->horizontalSlider_p, SIGNAL (valueChanged (int)), this, SLOT (pSliderValueChanged (int)));
  viewer->addPointCloud (cloud, "cloud");
  double RMSE = calculateRMSE(output, gt_disparity);
  ui->lcdNumber_RMSE->display(RMSE);
  double PSNR = calculatePSNR(output, gt_disparity);
  ui->lcdNumber_PSNR->display(PSNR);
  // viewer->setBackgroundColor (0, 0, 0);
  // viewer->addCoordinateSystem (1.0);
  // viewer->initCameraParameters ();

  pSliderValueChanged (2);
  viewer->resetCamera ();
  
  refreshView();
}

void
PCLViewer::onSliderReleased ()
{
  cv::Mat color = cv::imread("../data/Art/view1.png", 0);
	cv::Mat depth_low_res = cv::imread("../data/Art/lowres_disp1.png", 0);
  cv::Mat gt_disparity = cv::imread("../data/Art/disp1.png", 0);
  const int window_size = new_window_size;
  const float sigma = new_sigma;
  cv::Mat output = cv::Mat::zeros(color.rows, color.cols, CV_8UC1);
	Iterative_Upsampling(color, depth_low_res, output, window_size, sigma);

  const double focal_length = 3740;
  const double baseline = 0.160;

  // stereo estimation parameters
  const int dmin = 200;

  int height = output.size().height;
  int width = output.size().width;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  *cloud = Disparity2PointCloud_PCL(
    height, width, output,
    window_size, dmin, baseline, focal_length);
  double RMSE = calculateRMSE(output, gt_disparity);
  ui->lcdNumber_RMSE->display(RMSE);
  double PSNR = calculatePSNR(output, gt_disparity);
  ui->lcdNumber_PSNR->display(PSNR);
  viewer->updatePointCloud (cloud, "cloud");
  refreshView();
}

void
PCLViewer::pSliderValueChanged (int value)
{
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
  refreshView();
}

void
PCLViewer::refreshView()
{
#if VTK_MAJOR_VERSION > 8
  ui->qvtkWidget->renderWindow()->Render();
#else
  ui->qvtkWidget->update();
#endif
}

void
PCLViewer::sigmaSliderValueChanged (int value)
{
  new_sigma = value;
  printf ("sigmaSliderValueChanged: [%d]\n", new_sigma);
}

void
PCLViewer::windowSliderValueChanged (int value)
{
  new_window_size = value;
  printf ("windowSliderValueChanged: [%d]\n", new_window_size);
}

PCLViewer::~PCLViewer ()
{
  delete ui;
}