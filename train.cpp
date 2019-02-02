#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

int main(int argc, char *argv[]) {
  //Process image to extract contour
  cv::Mat thr, gray, con;
  cv::Mat src = cv::imread("digits.png",1);
  cv::cvtColor(src, gray, CV_BGR2GRAY);
  cv::threshold(gray ,thr, 200, 255, cv::THRESH_BINARY_INV); //Threshold to find contour
  thr.copyTo(con);

  // Create sample and label data
  std::vector< std::vector <cv::Point> > contours; // Vector for storing contour
  std::vector< cv::Vec4i > hierarchy;
  cv::Mat sample;
  cv::Mat response_array;
  cv::findContours( con, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); //Find contour

  for( int i = 0; i< contours.size(); i=hierarchy[i][0] ) // iterate through first hierarchy level contours
  {
    cv::Rect r= boundingRect(contours[i]); //Find bounding rect for each contour
    cv::rectangle(src, cv::Point(r.x,r.y), cv::Point(r.x+r.width,r.y+r.height), cv::Scalar(0,0,255),2,8,0);
    cv::Mat ROI = thr(r); //Crop the image
    cv::Mat tmp1, tmp2;
    cv::resize(ROI,tmp1, cv::Size(10,10), 0,0, cv::INTER_LINEAR ); //resize to 10X10
    tmp1.convertTo(tmp2, CV_32FC1); //convert to float
    sample.push_back(tmp2.reshape(1,1)); // Store  sample data
    cv::imshow("src",src);
    int c = cv::waitKey(0); // Read corresponding label for contour from keyoard
    c-=0x30;     // Convert ascii to intiger value
    response_array.push_back(c); // Store label to a mat
    cv::rectangle(src, cv::Point(r.x,r.y), cv::Point(r.x+r.width,r.y+r.height), cv::Scalar(0,255,0),2,8,0);
  }

  // Store the data to file
  cv::Mat response,tmp;
  tmp = response_array.reshape(1,1); //make continuous
  tmp.convertTo(response,CV_32FC1); // Convert  to float

  cv::FileStorage Data("DigitTrainingData.yml", cv::FileStorage::WRITE); // Store the sample data in a file
  Data << "data" << sample;
  Data.release();

  cv::FileStorage Label("DigitLabelData.yml", cv::FileStorage::WRITE); // Store the label data in a file
  Label << "label" << response;
  Label.release();
  std::cout<<"Training and Label data created successfully....!! "<<std::endl;

  cv::Ptr<cv::ml::TrainData>trainingData=cv::ml::TrainData::create(sample, cv::ml::SampleTypes::ROW_SAMPLE, response);
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->train(trainingData); // Train with sample and responses
  cv::FileStorage knnFS("TrainedKNN.knn", cv::FileStorage::WRITE);
  knn->write(knnFS);
  knnFS.release();

  cv::imshow("src",src);
  cv::waitKey();

  cv::waitKey(200);
  src = cv::imread("30_2.jpg", 1);
  cv::Mat planes[3];
  //Mat dst;
  cv::split(src,planes);
  cv::cvtColor(src, gray, CV_BGR2GRAY);

// BEGIN HOUGH CICLE PREPROCESSING
  cv::Mat result_hough;

  std::vector<cv::Vec3f> circles;

  /// Apply the Hough Transform to find the circles
  cv::HoughCircles( planes[2], circles, CV_HOUGH_GRADIENT, 1, gray.rows/8, 200, 100, 0, 0 );

  /// Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      cv::circle( src, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      cv::circle( src, center, radius, cv::Scalar(255,0,0), 3, 8, 0 );
   }

  /// Show your results
  cv::namedWindow( "Hough Circle Transform Demo", CV_WINDOW_NORMAL );
  cv::imshow( "Hough Circle Transform Demo", planes[0]);
  // center and radius are the results of HoughCircle
  // mask is a CV_8UC1 image with 0
  std::cout << (circles.size()) << std::endl;
  cv::Mat crop_result;
  if (circles.size()>0){
      cv::Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
      int radius = cvRound(circles[0][2]);
      radius = 0.9*radius;
      cv::Mat mask = cv::Mat::zeros( src.rows, src.cols, CV_8UC1 );
      cv::circle( mask, center, radius, cv::Scalar(255), -1); //-1 means filled
      gray.copyTo( crop_result, mask );
      cv::namedWindow("Cropped hough circle", CV_WINDOW_NORMAL);
      cv::imshow("Cropped hough circle", crop_result);
      cv::waitKey(0);
  } else {
    return 1;
  }

// CONTINUE NORMAL K MEANS
  cv::threshold(crop_result, thr,220,255, cv::THRESH_BINARY_INV); // Threshold to create input
  thr.copyTo(con);

  //Create input sample by contour finding and cropping
  cv::findContours( thr, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
  cv::Mat result = cv::Mat::zeros( src.rows, src.cols, CV_8UC3 );
  cv::Mat knn_result;

  std::cout << "Found " << contours.size() << " contours in image"<<std::endl;

  for( int i = 0; i< contours.size(); i++ ) // iterate through each contour for first hierarchy level .
  {
    cv::Rect r= boundingRect(contours[i]);
    cv::Mat ROI = crop_result(r);
    cv::Mat tmp1, tmp2;
    cv::resize(ROI,tmp1, cv::Size(10,10), 0, 0, cv::INTER_LINEAR );
    tmp1.convertTo(tmp2, CV_32FC1);
    float p = knn->findNearest(tmp2.reshape(1,1), 1, knn_result);
    char name[4];
    std::cout << "Found class "<<(int)p << std::endl;
    cv::drawContours(result, contours, i, cv::Scalar(255, 0, 0));
    cv::putText(result,name, cv::Point(r.x,r.y+r.height) ,0,1, cv::Scalar(0, 255, 0), 2, 8 );
  }

  cv::namedWindow("Found contours", CV_WINDOW_NORMAL);
  cv::imshow("Found contours", result);
  cv::imwrite("dest.jpg", result);
  cv::waitKey();
}
