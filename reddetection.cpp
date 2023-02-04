#include <opencv2/opencv.hpp>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
 
using namespace cv;
using namespace std;

// ------------ new functions ------------
// these should eventually be moved into its own class object (for optimization purposes) and then into its own header file library thing (for formatting purposes)

Point getCenterOfMat(Mat frame){ // for any given frame, calculates the center of it. Should only be ran once then stored!

     return Point(frame.cols*0.5, frame.rows*0.5);

}

Point getErrorFromScreenCenter(Point frame_center, Point target){ // dx & dy are calculated here in a Point object. Can be accessed with .x and .y functions

     return Point(target.x - frame_center.x, target.y - frame_center.y);

}

void drawCorrectionVector(Mat frame, Point frame_center, Point target, bool drawComponents = false){ // function for graphically notating information about the dx & dy

     //possible optimization - prereq the getErrorFromScreenCenter in a variable and reference it with a pointer.
     //                        in fact most things here can be dereferenced pointers instead of instantiated

     Point error = getErrorFromScreenCenter(frame_center, target);

     line(frame, target, frame_center, Scalar(255, 255, 255));

     if(drawComponents){
          line(frame, Point(frame_center.x + error.x, frame_center.y), getCenterOfMat(frame), Scalar(255, 0, 0));
          line(frame, Point(frame_center.x + error.x, frame_center.y + error.y), Point(frame_center.x + error.x, frame_center.y), Scalar(0, 255, 0));
     }
}

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! nvvidconv ! videobalance contrast=1 brightness =0 ! nvvidconv ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// -------------------------------------

int main(){

     //cpu mats
     Mat myCam, myCamHSV, red_mask;

     //gpu mats
     cuda::GpuMat gpu_myCam, gpu_myCamHSV, gpu_red_mask;

     // setting range for red
     Scalar red_lower = Scalar(170, 170, 50);
     Scalar red_upper = Scalar(180, 255, 255);

     // Creating the capture object for camera
     //VideoCapture cap(0);
     
     int capture_width = 1280 ;
     int capture_height = 720 ;
     int display_width = 1280 ;
     int display_height = 720 ;
     int framerate = 30 ;
     int flip_method = 0 ;

     std::string pipeline = gstreamer_pipeline(capture_width,
     capture_height,
     display_width,
     display_height,
     framerate,
     flip_method);
     std::cout << "Using pipeline: \n\t" << pipeline << "\n";

     cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

     while (true){
          // Capturing through camera
          // "myCam" is a single frame from the "cap" camera stream
          cap >> myCam;

          // myCam to HSV _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
          // cpu method
          //cvtColor(myCam, myCamHSV, COLOR_BGR2HSV);

          // gpu method
          gpu_myCam.upload(myCam); // loading myCam frame into VRAM
          cuda::cvtColor(gpu_myCam, gpu_myCamHSV, COLOR_BGR2HSV); // gpu cvtColor operation
          gpu_myCamHSV.download(myCamHSV); // dumping converted frame back into the cpu mat
     
          // making red mask
          inRange(myCamHSV, red_lower, red_upper, red_mask); // HSV -> red_mask
     
          Mat kernal = getStructuringElement(MORPH_RECT, Size(5, 5));
          dilate(red_mask, red_mask, kernal);
     
          // Finding contours of red object 
          vector<vector<Point> > contours_red;
          vector<Vec4i> hierarchy;
          findContours(red_mask, contours_red, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

          // everything black except for red
          // cpu method
          // Mat red;
          // bitwise_and(myCam, myCam, red, red_mask);

          // gpu method
          Mat red;
          cuda::GpuMat gpu_red;
          gpu_red_mask.upload(red_mask);
          cuda::bitwise_not(gpu_myCam, gpu_red, gpu_red_mask); // bitwise_and throws error, this is a workaround
          cuda::bitwise_not(gpu_red, gpu_red, gpu_red_mask); // https://github.com/opencv/opencv/issues/20698
          gpu_red.download(red);
          //gpu_red.release(); 
          gpu_red_mask.release(); // red_mask needs to be destroyed in VRAM or else it leaves residual masking data. uncomment and run to see what i mean

     
          // Tracking red color and drawing contours
          for (int i = 0; i < contours_red.size(); i++){
               double area = contourArea(contours_red[i]);
               if (area > 1500){
                    // Rectangles for red objects 
                    Rect rect = boundingRect(contours_red[i]);
                    rectangle(red, boundingRect(contours_red[i]), Scalar(255, 255, 255), 2);

                    Moments M = moments(contours_red[i]);
                    int cx, cy;

                    if (M.m00 != 0){
                         cx = (int) (M.m10 / M.m00);
                         cy = (int) (M.m01 / M.m00);
                         //drawContours(red, contours_red, i, Scalar(255, 255, 255), 2);
                         circle(red, Point(cx, cy), 7, Scalar(255, 255, 255), -1);
                         drawCorrectionVector(red, getCenterOfMat(red), Point(cx, cy), true);
                    }
               }
          }
          // Display the frame 
          imshow("Original", myCam);
          //imshow("Red", red); // show red only camera
     
          // Terminate the program if 'esc' is pressed
          if (waitKey(1) == 27){
               break;
          }
     }
     cap.release();
     return 0;
     } 