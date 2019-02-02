#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

Pix *mat8ToPix(cv::Mat *mat8)
{
    Pix *pixd = pixCreate(mat8->size().width, mat8->size().height, 8);
    for(int y=0; y<mat8->rows; y++) {
        for(int x=0; x<mat8->cols; x++) {
            pixSetPixel(pixd, x, y, (l_uint32) mat8->at<uchar>(y,x));
        }
    }
    return pixd;
}


void useTesseract(cv::Mat &img_in) {
    char *outText;

    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    // Open input image with leptonica library
//    cv::Mat src = cv::imread("10_2.jpg", 1);
    api->SetImage(img_in.data, img_in.cols, img_in.rows, 1, img_in.step);
    // Get OCR result
    outText = api->GetUTF8Text();
    printf("OCR output:\n%s", outText);

    // Destroy used object and release memory
    api->End();
    delete [] outText;
}

int main(int argc, char *argv[]) {
  cv::waitKey(200);
  cv::Mat thr, gray, con;
  cv::Mat src = cv::imread("10_2.jpg", 1);
  cv::cvtColor(src,gray, CV_BGR2GRAY);
  cv::threshold(gray,thr,200,255, cv::THRESH_BINARY_INV); // Threshold to create input
	thr.copyTo(con);

//  cv::Ptr<cv::ml::KNearest> knn = cv::Algorithm::load<cv::ml::KNearest>("TrainedKNN.knn");
//  knn->setIsClassifier(true);

  src = cv::imread("30_5.jpg", 1);
  cv::Mat planes[3];
  cv::cvtColor(src, src, CV_BGR2HSV);
  cv::Mat lower_red_hue_range, upper_red_hue_range;
  cv::inRange(src, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
  cv::inRange(src, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);
  cv::bitwise_or(lower_red_hue_range, upper_red_hue_range, planes[2]);
  std::vector<cv::Vec3f> circles;

  cv::namedWindow("In HSV range", CV_WINDOW_NORMAL);
  cv::imshow("In HSV range",  planes[2]);
  cv::waitKey(0);

  /// Apply the Hough Transform to find the circles
  cv::HoughCircles( planes[2], circles, CV_HOUGH_GRADIENT, 1, src.rows/8, 50, 25);

  /// Draw the circles detected
  cv::Mat draw_image = src.clone();
  for( size_t i = 0; i < circles.size(); i++ )
  {
      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      cv::circle( draw_image, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      cv::circle( draw_image, center, radius, cv::Scalar(255,0,0), 3, 8, 0 );
   }

  cv::cvtColor(src, src, CV_HSV2BGR);

  /// Show your results
  cv::namedWindow( "Hough Circle Transform Demo", CV_WINDOW_NORMAL );
  cv::imshow( "Hough Circle Transform Demo", draw_image);
  // center and radius are the results of HoughCircle
  // mask is a CV_8UC1 image with 0
  std::cout << "Number of cirles " << (circles.size()) << std::endl;
  cv::Mat crop_result;
  if (circles.size()>0){
      cv::Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
      int radius = cvRound(circles[0][2]);
      radius = 0.9*radius;
      cv::Mat mask = cv::Mat::zeros( src.rows, src.cols, CV_8UC1 );
      cv::circle( mask, center, radius, cv::Scalar(255), -1); //-1 means filled
      src.copyTo( crop_result, mask );
      cv::namedWindow("RED", CV_WINDOW_NORMAL);
      cv::imshow("RED", planes[2]);
      crop_result.setTo(cv::Scalar(0, 0, 0), planes[2] > 100);
      cv::cvtColor(crop_result, crop_result, CV_BGR2GRAY);
//      crop_result.setTo(cv::Scalar(255, 255, 255), crop_result == 0);
//      crop_result.setTo(cv::Scalar(1, 0, 0), crop_result < 80);
      crop_result.setTo(cv::Scalar(0, 0, 0), crop_result > 80);
      crop_result.setTo(cv::Scalar(255, 255, 255), crop_result > 0);
      cv::namedWindow("Cropped hough circle", CV_WINDOW_NORMAL);
      cv::imshow("Cropped hough circle", crop_result);
      cv::Mat element = cv::getStructuringElement(2, cv::Size(5, 5));
      cv::morphologyEx(crop_result, crop_result, cv::MORPH_OPEN, element);
      cv::waitKey(0);
  } else {
    return 1;
  }

  useTesseract(crop_result);

//  std::vector< std::vector <cv::Point> > contours; // Vector for storing contour
//  std::vector< cv::Vec4i > hierarchy;

//  //Create input sample by contour finding and cropping
//  cv::findContours( con, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
//  cv::Mat dst(src.rows, src.cols, CV_8UC3, cv::Scalar::all(0));
//  cv::Mat result;

//  for( int i = 0; i< contours.size(); i=hierarchy[i][0] ) // iterate through each contour for first hierarchy level .
//  {
//    cv::Rect r= boundingRect(contours[i]);
//    cv::Mat ROI = thr(r);
//    cv::Mat tmp1, tmp2;
//    cv::resize(ROI,tmp1, cv::Size(10,10), 0, 0, cv::INTER_LINEAR );
//    tmp1.convertTo(tmp2, CV_32FC1);
//    float p = knn->findNearest(tmp2.reshape(1,1), 1, result);
//    char name[4];
//    sprintf(name,"%d",(int)p);
//    cv::putText(dst,name, cv::Point(r.x,r.y+r.height) ,0,1, cv::Scalar(0, 255, 0), 2, 8 );
//  }

//  cv::imshow("src", src);
//  cv::imshow("dst", dst);
//  cv::imwrite("dest.jpg", dst);
//  cv::waitKey();
}
