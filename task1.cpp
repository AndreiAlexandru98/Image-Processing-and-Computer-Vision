/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - task1.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "resources/frontalface.xml";
CascadeClassifier cascade;



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


/** @function main */
int main( int argc, const char** argv ) {
  // Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat frame_gray;
	int nFaces = 0, truePositives = 0, falseNegatives = 0, falsePositives = 0;
	float f1Score = 0, tpr = 0;
	std::vector<Rect> faces;
	string line, source;

	// Load the Strong Classiffaces.size(ier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	ifstream in("resources/faces.txt");
	if (in) {
		while (getline(in, line)) {
        //Get the source file
				source = extractFirst(line);
				line = extractLast(line);

        //Get the number of faces for each image
				nFaces = atoi(extractFirst(line).c_str());

				line = extractLast(line);
			  int truth[nFaces][4];
				//Check which is the current image
				if(source == argv[1] ) {
             // Store the face parameters
					   for(int i = 1; i <= nFaces; i++) {
							  for(int j = 0; j < 4;j++) {
							     truth[i-1][j] = atoi(extractFirst(line).c_str());
									 line = extractLast(line);
								 }
						  }

						  // Draw bounding boxes for ground truths
              for(int i = 0; i < nFaces; i++) {
                rectangle(frame, Point(truth[i][0],truth[i][1]), Point(truth[i][0] + truth[i][2],truth[i][1] + truth[i][3]), Scalar( 0, 0, 255 ), 2);
						  }

						  // Prepare image by turning it into Grayscale and normalising lighting
						  cvtColor( frame, frame_gray, CV_BGR2GRAY );
						  equalizeHist( frame_gray, frame_gray );

						  //Perfom Perform Viola-Jones Object Detection
							cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
              // Draw bounding boxes for detected faces

							for( int i = 0; i < faces.size(); i++ ) {
								rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
							}

              // Calculate truePositives
							for( int i = 0; i < faces.size(); i++ ){
								 for(int j = 0; j < nFaces; j++) {
									 if(abs(faces[i].x - truth[j][0]) < 17*truth[j][0] / 100 &&
											abs(faces[i].y - truth[j][1])  < 17*truth[j][1] / 100  &&
											abs(faces[i].height -truth[j][3]) < 17*truth[j][3] / 100 ) {
											truePositives++;
										}
									}
							 }

              // Calculate true positive rate
							falseNegatives = nFaces - truePositives;
							if(truePositives == 0) {
							  tpr = 0;
						  }
							else {
							  tpr = (float)truePositives / ((float)truePositives + (float)falseNegatives);
						  }
							//Uncomment to display TRUE POSITIVE RATE
              //cout << "True positive rate is " << tpr << "\n";

							// Calculate F1 score
							falsePositives = faces.size() - nFaces;
							if(truePositives == 0 && falsePositives == 0 && falseNegatives == 0) {
							  f1Score = 1;
							}
							else {
							  f1Score = 2 * (float)truePositives / (2 * (float)truePositives + (float)falsePositives + (float)falseNegatives);
							}
							//Uncomment to display F1 SCORE
							//cout << "F1 score is " << f1Score << "\n";
						}
		}
  }

	// Display image all bounding boxes
	imwrite( "detected.jpg", frame );
	return 0;
}
