#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
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
#include "pclviewer.h"
#include <QApplication>
#include <QMainWindow>

using namespace std::chrono_literals;

pcl::visualization::PCLVisualizer::Ptr normalsVis (
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

pcl::visualization::PCLVisualizer::Ptr cloudsVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

pcl::visualization::PCLVisualizer::Ptr trianglesVis (
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PolygonMesh triangles)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addPolygonMesh(triangles, "sample triangles");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

void Disparity2PointCloud_PCL(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  std::stringstream out3d;
  out3d << output_file << ".xyz";
  std::ofstream outfile(out3d.str());

  std::stringstream out3d_pcd;
  out3d_pcd << output_file << ".pcd";

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
      outfile << X << " " << Y << " " << Z << std::endl;
    }
  }
  cloud.width = cloud.size (); cloud.height = 1; cloud.is_dense = true;
  pcl::io::savePCDFileASCII(out3d_pcd.str(), cloud);
  std::cerr << "Saved " << cloud.size () << " data points to " << out3d_pcd.str() << std::endl;
}


void Display_PointCloud(const std::string& output_file)
{
  std::stringstream out3d;
  out3d << output_file << ".pcd";
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(out3d.str(), *cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered);


  // https://pcl.readthedocs.io/projects/tutorials/en/latest/pcl_visualizer.html
  // https://pointclouds.org/documentation/tutorials/greedy_projection.html#compiling-and-running-the-program

  pcl::visualization::PCLVisualizer::Ptr viewer;

  std::cout << "STARTING NORMALS COMPUTATION" << std::endl;
  // normal estimation
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (cloud_filtered);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  tree->setInputCloud(cloud_filtered);
  ne.setSearchMethod (tree);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch (0.1);
  // ne.setKSearch (20);
  ne.compute (*cloud_normals);

  std::cout << "NORMALS ARE COMPUTED" << std::endl;

  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*cloud_filtered, *cloud_normals, *cloud_with_normals);

   // Create search tree*
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_with_normals);

  // Initialize objects
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;

  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (0.1);

  // Set typical values for the parameters
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (150);
  gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
  gp3.setMinimumAngle(M_PI/18); // 10 degrees
  gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
  gp3.setNormalConsistency(false);

  // Get result
  gp3.setInputCloud (cloud_with_normals);
  gp3.setSearchMethod (tree2);
  gp3.reconstruct (triangles);


  std::cout << "COMPUTED TRIANGLES" << std::endl;

  viewer = cloudsVis(cloud_filtered);
  viewer = normalsVis(cloud_filtered, cloud_normals);
  viewer = trianglesVis(cloud, triangles);

  // BUG in my environment? The last window closes so added a 4th arbitrary window
  viewer = cloudsVis(cloud_filtered);


  std::stringstream out3d_obj;
  out3d_obj << output_file << "_mesh.obj";

  pcl::io::saveOBJFile (out3d_obj.str(), triangles);
  
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    // std::this_thread::sleep_for(100ms);
  }
}

int main(int argc, char** argv) {

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 0.160;

  // stereo estimation parameters
  const int dmin = 200;
  const int window_size = 5;

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " DISPARITY_IMAGE OUTPUT_FILE" << std::endl;
    return 1;
  }

  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  const std::string output_file = argv[2];

  if (!image1.data) {
    std::cerr << "No disparity image data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size = " << window_size << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "output filename = " << argv[2] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;

  ////////////
  // Output //
  ////////////

  Disparity2PointCloud_PCL(
    output_file,
    height, width, image1,
    window_size, dmin, baseline, focal_length);
  
  Display_PointCloud(output_file);

  return 0;
}
