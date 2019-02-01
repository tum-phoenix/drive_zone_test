#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

Mat* imtrans(Mat* img, Mat* r){
  Mat src,src_gray;

  /// Read the image
  src = imread( "/data/Images/Zonen/50_3.jpg", 1 );
  src=*img;
  if( !src.data )
    { return NULL; }
  //Mat src=imread("my.png");
  Mat planes[3];
  //Mat dst;
  split(src,planes);
  /// Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Reduce the noise so we avoid false circle detection
  //GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

  vector<Vec3f> circles;

  /// Apply the Hough Transform to find the circles
  HoughCircles( planes[2], circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );

  /// Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( src, center, radius, Scalar(255,0,0), 3, 8, 0 );
   }

  /// Show your results
  namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
//imshow( "Hough Circle Transform Demo", src );
  imshow( "Hough Circle Transform Demo", planes[0] );
  // center and radius are the results of HoughCircle
  // mask is a CV_8UC1 image with 0
  cout << (circles.size()) << endl;
  if (circles.size()>0){
      Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
      int radius = cvRound(circles[0][2]);
      Mat mask = Mat::zeros( src.rows, src.cols, CV_8UC1 );
      Mat dst=Mat(src.rows, src.cols, CV_8UC3, Scalar(255,255,255));
      Mat roi( src, Rect( center.x-radius, center.y-radius, radius*2, radius*2 ) );

      circle( mask, center, radius, Scalar(255,255,255), -1, 8, 0 ); //-1 means filled
      src.copyTo( dst, mask ); // copy values of img to dst if mask is > 0.
      Mat binaryMat(src.size(), src.type());
         //Apply thresholding
      threshold(dst, binaryMat, 150, 255, cv::THRESH_BINARY);
      *r=binaryMat;
      imshow( "Hough Circle Transform Demo", src);
      waitKey(0);
  }
  imshow( "Hough Circle Transform Demo", planes[2]);
  waitKey(0);

  return r;

}


Mat* data(){
    Mat zone10 = imread( "/home/basti/drive_ml_new/sign_recognition/images/10_zone_beginn.png", 1 );
    Mat zone20 = imread( "/home/basti/drive_ml_new/sign_recognition/images/20_zone_beginn.png", 1 );
    Mat zone30 = imread( "/home/basti/drive_ml_new/sign_recognition/images/30_zone_beginn.png", 1 );
    Mat zone40 = imread( "/home/basti/drive_ml_new/sign_recognition/images/40_zone_beginn.png", 1 );
    Mat zone50 = imread( "/home/basti/drive_ml_new/sign_recognition/images/50_zone_beginn.png", 1 );
    Mat zone60 = imread( "/home/basti/drive_ml_new/sign_recognition/images/60_zone_beginn.png", 1 );
    Mat zone70 = imread( "/home/basti/drive_ml_new/sign_recognition/images/70_zone_beginn.png", 1 );
    Mat zone80 = imread( "/home/basti/drive_ml_new/sign_recognition/images/80_zone_beginn.png", 1 );
    Mat zone90 = imread( "/home/basti/drive_ml_new/sign_recognition/images/90_zone_beginn.png", 1 );
    Mat zone10_b;
    imtrans(&zone20,&zone10_b);
}



int main()
{
    Mat src = imread( "/data/Images/Zonen/40_2.jpg", 1 );
    //imtrans(&src);
    data();
    cout << "Hello World!" << endl;
    return 0;

}


