#pragma once

#include <iostream>

// Qt
#include <QMainWindow>

// Point Cloud Library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui
{
  class PCLViewer;
}

class PCLViewer : public QMainWindow
{
  Q_OBJECT

public:
  explicit PCLViewer (QWidget *parent = 0);
  ~PCLViewer ();

public Q_SLOTS:

  void
  onSliderReleased ();

  void
  pSliderValueChanged (int value);

  void
  sigmaSliderValueChanged (int value);

  void
  windowSliderValueChanged (int value);

protected:
  void
  refreshView();

  pcl::visualization::PCLVisualizer::Ptr viewer;
  PointCloudT::Ptr cloud;

  unsigned int new_window_size;
  unsigned int new_sigma;

private:
  Ui::PCLViewer *ui;
};