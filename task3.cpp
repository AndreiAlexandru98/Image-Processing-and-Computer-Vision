// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


using namespace cv;
using namespace std;

/** Global variables */
String cascade_name = "resources/dart.xml";
CascadeClassifier cascade;

int houghDataSize;

int *xsHough;
int *ysHough;
int *rsHough;

//Get the remaining string after resizing
string extractLast(string source){

	for(int i=0; i<source.length(); i++)
		if(source[i] == ' ') {
				source.erase(0,i + 1);
				break;
		}

	return source;
}

//Get the first word
string extractFirst(string first) {

	for(int i=0; i<first.length(); i++)
		if(first[i] == ' ') {
			first.resize(i);
			break;
		}

	return first;
}

//Initializing the 3D array
int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {

        array[i] = (int **) malloc(dim2 * sizeof(int *));
	for (j = 0; j < dim2; j++) {
  	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
	}

    }
    return array;
}

//Set all array values to 0
void initZero(int ***array, int x, int y, int z){

  for(int i = 0; i <x; i++)
    for(int j = 0; j <y; j++)
      for(int g = 0; g<z; g++)
        array[i][j][g] = 0;


}


//Calculate the x gradient component
float xGradient(Mat image, int x, int y) {

 return image.at<float>(y-1, x-1) +
	       2*image.at<float>(y, x-1) +
	        image.at<float>(y+1, x-1) -
	         image.at<float>(y-1, x+1) -
	          2*image.at<float>(y, x+1) -
	           image.at<float>(y+1, x+1);
}

//Calculate the y gradient component
float yGradient(Mat image, int x, int y) {

 return image.at<float>(y-1, x-1) +
         2*image.at<float>(y-1, x) +
          image.at<float>(y-1, x+1) -
           image.at<float>(y+1, x-1) -
            2*image.at<float>(y+1, x) -
             image.at<float>(y+1, x+1);
}

//Calculate the hough circles
void hough(Mat mag_thr,Mat dir, Mat image){
 	int x0, y0, x1, y1;
 	int theta;
  int rows, cols, radius;

  rows =  mag_thr.rows;
  cols = mag_thr.cols;
  radius = rows > cols ? rows/2 : cols/2;

	int ***space3D = malloc3dArray(rows , cols, radius);
	initZero(space3D, rows , cols, radius);


	for(int y = 1; y < rows; y++){
		for(int x = 1; x < cols; x++){
			//check the threshold
			if(mag_thr.at<int>(y,x) > 120) {
        float thetaF = dir.at<float>(y,x);
        for(int r = 30; r <= radius; r ++) {
					//Apply small variance for theta
          for (int i = -2; i < 2; i++) {

           float thetaWithSmallAmountOfVariance = thetaF + (((float)i) / 1000.f);
           //Calculate the new coordonates
           x0 = x + r * cos(thetaWithSmallAmountOfVariance);
			     y0 = y + r * sin(thetaWithSmallAmountOfVariance);

           x1 = x - r * cos(thetaWithSmallAmountOfVariance);
           y1 = y - r * sin(thetaWithSmallAmountOfVariance);


           //Start voting
           if(y1 >= 0 && x1 >=0 && y1<rows && x1<cols)
            space3D[y1][x1][r] += 1;

           if(y0 >= 0 && x0 >=0 && y0<rows && x0<cols)
            space3D[y0][x0][r] += 1;
           }
		     }
      }
    }
  }


  //Create 2D image to display hough space
  Mat oof = mag_thr.clone();

	//Set all matrix values to 0
  for(int y = 0; y < rows; y++){
  	for(int x = 0; x < cols; x++){

      oof.at<float>(y, x) = 0;
    }
	}


  int maxR = 0;
  int maxVal = 0;

  int xs[cols];
  int ys[rows];
  int rs[radius];

  xsHough = new int[cols];
  ysHough = new int[rows];
  rsHough = new int[radius];

  int i = 0;

	//Store the number of votes in the 2D image
  for(int y = 1; y < rows; y++) {
   for(int x = 1; x < cols; x++) {
     for(int r = 30; r <= 120; r ++) {
       oof.at<float>(y, x) += space3D[y][x][r];

			 //Store the r value for each (x,y) coordonates
       if(space3D[y][x][r] > maxVal){
         maxR = r;
         maxVal = space3D[y][x][r];
       }
      }

     if(space3D[y][x][maxR] > maxR/0.75) {
        Point center(x, y);
        bool checkNeighbours = false;
				//Check for neighbours
        for(int j = 0; j < i; j++) {
          if(abs(x - xs[j]) < 30 && abs(y - ys[j]) < 30) {
            checkNeighbours = true;
					}
        }
				//Store coordonates
        if(checkNeighbours != true) {
         xs[i] = x;
         ys[i] = y;
         rs[i] = maxR;
         xsHough[i] = x - (int)maxR;
         ysHough[i] = y - (int)maxR;
         rsHough[i] = maxR;
         i++;
        }
      }
    }
  }

houghDataSize = i;
//Display 2D hough image
imwrite( "hough.jpg", oof );
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//MAIN FUNCTION
int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 image.convertTo(image, CV_32FC1);
 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }


Mat gray_image;
Mat g;
cvtColor( image, g, CV_BGR2GRAY );
GaussianBlur(g, gray_image, Size(3, 3), 0, 0);


//SOBEL
Mat xDirection= gray_image.clone();
Mat yDirection= gray_image.clone();
Mat mag = gray_image.clone();
Mat dir = gray_image.clone();
float gx,gy;
for(int y = 1; y < xDirection.rows;y++){
	for(int x = 1; x < xDirection.cols;x++){
          gx = xGradient(gray_image,x,y);
					gy = yGradient(gray_image,x,y);
					xDirection.at<float>(y,x) = gx;
					yDirection.at<float>(y,x) = gy;
					mag.at<float>(y,x) = sqrt(gx * gx + gy * gy);
					dir.at<float>(y,x) = atan2(gy,gx);

				//	dir.at<float>(y,x) = dir.at<float>(y,x)> 255 ? 255:dir.at<float>(y,x);
				//	dir.at<float>(y,x) = dir.at<float>(y,x) < 0 ? 0 : dir.at<float>(y,x);

	}
}
//THRESHOLDING
Mat mag_thr = mag.clone();
for(int y = 1; y < xDirection.rows;y++){
	for(int x = 1; x < xDirection.cols;x++){
         if(mag.at<float>(y,x) > 135)
				        mag_thr.at<float>(y,x) = 255;
				else
				        mag_thr.at<float>(y,x) = 0;

	}
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Read Input Image
Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
Mat frame_gray;
int nDarts = 0, truePositives = 0, falseNegatives = 0, falsePositives = 0;
float f1Score = 0, tpr = 0;
std::vector<Rect> darts;
string line, source;

////////////////////////////////////////////////////////////////
hough(mag_thr, dir, frame);
////////////////////////////////////////////////////////////////


// Load the Strong Classiffaces.size(ier in a structure called `Cascade'
if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

ifstream in("resources/darts.txt");
if (in) {
  while (getline(in, line)) {
      //Get the source file
      source = extractFirst(line);
      line = extractLast(line);

      //Get the number of faces for each image
      nDarts = atoi(extractFirst(line).c_str());

      line = extractLast(line);
      int truth[nDarts][4];
      //Check which is the current image
      if(source == argv[1] ) {
           // Store the face parameters
           for(int i = 1; i <= nDarts; i++) {
              for(int j = 0; j < 4;j++) {
                 truth[i-1][j] = atoi(extractFirst(line).c_str());
                 line = extractLast(line);
               }
            }

            // Draw bounding boxes for ground truths
            for(int i = 0; i < nDarts; i++) {
              rectangle(frame, Point(truth[i][0],truth[i][1]), Point(truth[i][0] + truth[i][2],truth[i][1] + truth[i][3]), Scalar( 0, 0, 255 ), 2);
            }

            // Prepare image by turning it into Grayscale and normalising lighting
            cvtColor( frame, frame_gray, CV_BGR2GRAY );
            equalizeHist( frame_gray, frame_gray );

            //Perfom Perform Viola-Jones Object Detection
            cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );


            int counter = 0;
            int finalSize = darts.size();
            bool checkNeighbours = false;
            int finalCords[finalSize][4];

            // Draw bounding boxes for detected faces
            for( int i = 0; i < darts.size(); i++ ) {
              for(int j = 0; j < houghDataSize; j++){
                if(abs(xsHough[j] - darts[i].x) < 40 &&
                      abs(ysHough[j] - darts[i].y) < 40) {


                    for(int k = 0; k < counter; k ++) {
                      if(abs(finalCords[k][0] - darts[i].x) < 30 && abs(finalCords[k][1] - darts[i].y) < 30 ) {
                        checkNeighbours = true;
											}
										}

										if(checkNeighbours != true) {
											//Draw boxes for the detected dartboards
                      rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
										  //Save box coordonates
                      finalCords[counter][0] = darts[i].x;
                      finalCords[counter][1] = darts[i].y;
                      finalCords[counter][2] = darts[i].width;
                      finalCords[counter][3] = darts[i].height;
                      //Count the number of boxes
                      counter++;
									  }
                 }
               }
             }
						 //CASE: No boxes were drawn
             if(counter == 0) {
               for(int i = 0; i < houghDataSize; i++) {
							   //Draw boxes
                 rectangle(frame, Point(xsHough[i], ysHough[i]), Point(xsHough[i] + rsHough[i]*2, ysHough[i] + rsHough[i]*2), Scalar( 0, 255, 0 ), 2);
								 //Save coordonates
                 finalCords[counter][0] = xsHough[i];
                 finalCords[counter][1] = ysHough[i];
                 finalCords[counter][2] = rsHough[i]*2;
                 finalCords[counter][3] = rsHough[i]*2;
								 //Count the number of boxes
                 counter++;
                }
            }

            finalSize = counter;


            // Calculate truePositives
            for( int i = 0; i < darts.size(); i++ ){
               for(int j = 0; j < counter; j++) {
                 if(abs(finalCords[i][0] - truth[j][0]) < 10*truth[j][0] / 100 &&
                    abs(finalCords[i][1]- truth[j][1])  < 10*truth[j][1] / 100  &&
                    abs(finalCords[i][2] -truth[j][3]) < 10*truth[j][3] / 100 ) {
                    truePositives++;
                  }
                }
             }

            // Calculate true positive rate
            falseNegatives = nDarts - truePositives;
            if(truePositives == 0) {
              tpr = 0;
            }
            else {
              tpr = (float)truePositives / ((float)truePositives + (float)falseNegatives);
            }
						//Uncomment to display TRUE POSITIVE RATE
            //cout << "True positive rate is " << tpr << "\n";

            // Calculate F1 score
            falsePositives = counter - nDarts;
            if(truePositives == 0 && falsePositives == 0 && falseNegatives == 0) {
              f1Score = 1;
            }
            else if((2 * (float)truePositives + (float)falsePositives + (float)falseNegatives) == 0)
            f1Score = 0;

            else{
              f1Score = 2 * (float)truePositives / (2 * (float)truePositives + (float)falsePositives + (float)falseNegatives);
            }
						//Uncomment to display F1 SCORE
            //cout << "F1 score is " << f1Score << "\n";
          }
  }
}

//Display the thresholded magnitude_thr
imwrite( "magnitude_thr.jpg", mag_thr );
//Display detections
imwrite( "detected.jpg", frame );
 return 0;
}
