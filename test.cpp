#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;


int main(int argc, char **argv){
  Mat frame;
  VideoCapture cap;
  int deviceID=0;
  int apiID=cv::CAP_ANY;
  cap.open(deviceID, apiID);
   if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

	Mat img_blob = frame.clone ();
	Mat new_img = frame.clone();//Mat(frame.rows, frame.cols, CV_64FC3, Scalar(0,0,0));
	std::vector < KeyPoint > keypoints;

	for(int i=0; i<img_blob.rows; i++){
	  for(int j=0; j<img_blob.cols; j++){
	    Scalar intensity = img_blob.at<Vec3b>(i,j);
	    int B = intensity[0];
	    int G = intensity[1];
	    int R = intensity[2];
	    
	    //if((B-R>30 && B-G>20)||(G<2&&B<2)){
	    if(fabs(R-180)<20 && fabs(B-180)<25 && fabs(G-180)<25){
	    //if(B<70 && R<70 &&G<70){
	      
	      new_img.at<Vec3b>(i,j)[0]=0;
		new_img.at<Vec3b>(i,j)[1]=255;
	      new_img.at<Vec3b>(i,j)[2]=255;

	    }
	  }
	  
	  
	}
	
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	
	//detect white
	//params.filterByColor=true;  
	params.blobColor = 0;//255;
	
	// Change thresholds
	params.minThreshold = 80;
	params.maxThreshold = 255;
  
	// Filter by Area.
	params.filterByArea = 1;
	params.minArea = 10;
	params.maxArea = 10000;
  
	// Filter by Circularity
	params.filterByCircularity = 1;
	params.minCircularity = 0.0;

	//Filter by distance
	params.minDistBetweenBlobs = 10;
	// Filter by Convexity
	params.filterByConvexity = 0;
	params.minConvexity = 0;
  
	// Filter by Inertia
	params.filterByInertia = 1;
	params.minInertiaRatio = 0.0;
  
	// Set up the detector with set parameters.                                                                                         
	Ptr < SimpleBlobDetector > detector = SimpleBlobDetector::create (params);
  
	// Detect blobs.                                                                                                                        
	detector->detect (img_blob, keypoints);
	///////////////////////////////
	
	  Mat img_flt = img_blob.clone ();
	  int d = 6;	// value 5-9 distance around each pixel to filter (must be odd)
	    int sigColor = 100;	// range of colours to call the same
	    int sigSpace = 50;	// ???
	    
	    bilateralFilter (img_blob, img_flt, d, sigColor, sigSpace);
	    imshow("bilateral",img_flt);
	//////////////////////////////
	    
	Mat grayscaleImage;
        cvtColor(img_blob, grayscaleImage, COLOR_BGR2GRAY);
	cv::imshow("grey",grayscaleImage);
	for (double thresh = 80; thresh < 150; thresh += 1){
	Mat binarizedImage;
	threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);
	cv::imshow("binary", binarizedImage);
	threshold(grayscaleImage, binarizedImage, thresh, 255, ADAPTIVE_THRESH_MEAN_C);//THRESH_BINARY);
	  
	cv::imshow("binary adaptmen", binarizedImage);
	threshold(grayscaleImage, binarizedImage, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C);
	 cv::imshow("binaryGaus", binarizedImage);
	//cv::imshow("grey2",grayscaleImage);
	}
	//blob vector will contain x,y,r
	std::vector < KeyPoint > kp;
	vector < Vec3f > blobs;
	for (KeyPoint keypoint:keypoints) {
	  //Point center1 = keypoint.pt;
	  int x = keypoint.pt.x;
	  int y = keypoint.pt.y;
	  float r = ((keypoint.size) + 0.0) / 2;
	 
	  Vec3f temp;
	  temp[0] = x;
	  temp[1] = y;
	  temp[2] = r;
	  
	  //blue -red>50 blue-green>30
	  cv::Scalar inten = frame.at<uchar>(y,x);
	  int B = inten.val[0];
	  int G = inten.val[1];
	  int R = inten.val[2];
	  bool blue=false;
	  if(B-R>50 && B-G>30 && B>200){blue=true;}
	    
	  if(blue){
	    kp.push_back(keypoint);
	  }
	  blobs.push_back (temp);  
	}
	
	//draw_found_center (blobs, blob_circles);
	//blob_circles = Mat::zeros (image_clahe.size (), image_clahe.type ());
	//draw_foundblobs( blobs, blob_circles );
	
	
	// show live and wait for a key with timeout long enough to show images
	drawKeypoints( frame, keypoints, img_blob, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	cv::imshow("Live", img_blob);
	cv::imshow("Threshold", new_img);
	if (waitKey(5) >= 0)
            break;
    }

  return 0;
}
