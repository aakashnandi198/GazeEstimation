#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include<string.h>

using namespace cv;
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

 //result file handler values
 FILE *res;

// blue bounding box variable
 int oboxx=187;
 int oboxy=113;
 int oboxw=261;
 int oboxh=270;

 int gap=22;

 int iboxx=oboxx+gap;
 int iboxy=oboxy+gap;
 int iboxw=oboxw-2*gap;
 int iboxh=oboxh-2*gap;

 int poscor=0;  //tell whether the face is well within the bounding box or not
 int clkcnt=0;

Mat debugImage;

//anthropometric ratios

int kEyePercentTop = 30;          //25     eyebrowrem 22
int kEyePercentSide = 20;         //13      eyebrowrem 13
int kEyePercentHeight = 16;    //30     eyebrowrem 11
int kEyePercentWidth = 20;      //35        eyebrowrem 31

//plus sign size and pupil coordinates
const int tada=5;
int ual1=37,lal1=27;
Point leftPupil;
Point rightPupil;
Point wleftPupil;
Point wrightPupil;
int ylefteyebrow;
int yrighteyebrow;


// matrices
Mat gradientX,gradientY;
Mat weight;

//postprocessing
const float kPostProcessThreshold = 0.98;  //0.97

//algorithm parameter
const int kFastEyeWidth = 50;
const float kWeightDivisor = 150.0;
const double kGradientThreshold = 0.3;
const int kWeightBlurSize = 5;

float euclidean(int a,int b,int c,int d)
{
    int l1,l2,sum;
    float dist1;
    l1=(a-c)*(a-c);
    l2=(b-d)*(b-d);
    sum=l1+l2;
    dist1=sqrtf(sum);
    return dist1;
}

Mat computeMatXGradient(const Mat &mat)
	{
	Mat out(mat.rows,mat.cols,CV_64F);
	for (int y = 0; y < mat.rows; ++y)
		{
		const uchar *Mr = mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);
		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < mat.cols - 1; ++x)
			{
			Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
			}
		Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
		}
	return out;
	}

double computeDynamicThreshold(const Mat &mat, double stdDevFactor)
	{
	Scalar stdMagnGrad, meanMagnGrad;
	meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
	}

void scaleToFastSize(const Mat &src,Mat &dst)
	{
	resize(src, dst, Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
	}

void testPossibleCentersFormula(int x, int y,double gx, double gy, Mat &out)
	{
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy)
		{
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);
		double *Or = out.ptr<double>(cy);
		for (int cx = 0; cx < out.cols; ++cx)
			{
			if (x == cx && y == cy)
				{
				continue;
				}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx*gx + dy*gy;
			dotProduct = std::max(0.0,dotProduct);  //ensure that negative value is set to zero before squaring
			// square and multiply by the weight
            //  if((euclidean(x,y,cx,cy)*255)/euclidean(0,0,weight.rows,weight.cols)>lal1 &&(euclidean(x,y,cx,cy)*255)/euclidean(0,0,weight.rows,weight.cols)<ual1)
            if((euclidean(x,y,cx,cy)*255)/euclidean(0,0,weight.rows,weight.cols)>lal1 &&(euclidean(x,y,cx,cy)*255)/euclidean(0,0,weight.rows,weight.cols)<ual1)
            Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
			}
		}
}

Point unscalePoint(Point p, Rect origSize)
	{
	float ratio = (((float)kFastEyeWidth)/origSize.width);
	int x = round(p.x / ratio);
	int y = round(p.y / ratio);
	return Point(x,y);
	}

bool inMat(Point p,int rows,int cols)     //checks whether a given point is in the matrix or not
	{
	return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
	}

bool floodShouldPushPoint(const Point &np, const Mat &mat)
	{
	return inMat(np, mat.rows, mat.cols);
	}


Mat floodKillEdges(Mat &mat)
	{
	rectangle(mat,Rect(0,0,mat.cols,mat.rows),255);
	Mat mask(mat.rows, mat.cols, CV_8U, 255);
	std::queue<Point> toDo;
	toDo.push(Point(0,0));
	while (!toDo.empty())
		{
		Point p = toDo.front();
		toDo.pop();
		if (mat.at<float>(p) == 0.0f)
			{
			    mask.at<uchar>(p)=0;
                continue;
			}
		// add in every direction
		Point np(p.x + 1, p.y); // right
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x - 1; np.y = p.y; // left
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y + 1; // down
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y - 1; // up
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		// kill it
		mat.at<float>(p) = 0.0f;
		mask.at<uchar>(p) = 0;
		}
		Point p;
    for(int x1=0;x1<mat.cols;x1++)
    for(int y1=0;y1<mat.rows;y1++)
    {
        p.x=x1;
        p.y=y1;
        if(mat.at<float>(p)!=0.0f)
            {
                mask.at<uchar>(p)=255;
            }
            else
            {
                mask.at<uchar>(p)=0;
            }
    }
	return mask;
    }

Mat matrixMagnitude(const Mat &matX, const Mat &matY)
	{
	Mat mags(matX.rows,matX.cols,CV_64F);
	for (int y = 0; y < matX.rows; ++y)
		{
		const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < matX.cols; ++x)
		 	{
			double gX = Xr[x], gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		 	}
		}
	return mags;
	}


Point findEyeCenter(Mat face,Rect eye, std::string debugWindow)
	{
	Mat eyeROIUnscaled = face(eye);
	Mat eyeROI;
	scaleToFastSize(eyeROIUnscaled, eyeROI);

	// draw eye region
	rectangle(face,eye,230);
	//-- Find the gradient
	gradientX = computeMatXGradient(eyeROI);

	gradientY = computeMatXGradient(eyeROI.t()).t();

	//-- Normalize and threshold the gradient
	// compute all the magnitudes
	Mat mags = matrixMagnitude(gradientX, gradientY);

	//compute the threshold
	double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);

	//normalize
	for (int y = 0; y < eyeROI.rows; ++y)
		{
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < eyeROI.cols; ++x)
			{
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];
			if (magnitude > gradientThresh)
				{
				Xr[x] = gX/magnitude;
				Yr[x] = gY/magnitude;
				}
			else
				{
				Xr[x] = 0.0;
				Yr[x] = 0.0;
				}
			}
		}

	//-- Create a blurred and inverted image for weighting

	GaussianBlur( eyeROI, weight,Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
	for (int y = 0; y < weight.rows; ++y)
		{
		unsigned char *row = weight.ptr<unsigned char>(y);
		for (int x = 0; x < weight.cols; ++x)
			{
			row[x] = (255 - row[x]);
			}
		}

	//-- Run the algorithm!
	Mat outSum = Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
	// for each possible gradient location
	// Note: these loops are reversed from the way the paper does them
	// it evaluates every possible center for each gradient location instead of
	// every possible gradient location for every center.

	for (int y = 0; y < weight.rows; ++y)
		{
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = 0; x < weight.cols; ++x)
			{
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0)
				{
				continue;
				}
			testPossibleCentersFormula(x, y, gX, gY, outSum);                                                                            //bug in weight. please check the weight passed.
			}
		}

	double numGradients = (weight.rows*weight.cols);
	Mat out;
	outSum.convertTo(out, CV_32F,1.0/numGradients);

	//-- Find the maximum point
	Point maxP;
	double maxVal;
	minMaxLoc(out, NULL,&maxVal,NULL,&maxP);

	//-- Flood fill the edges
    Mat floodClone;
    double floodThresh = maxVal * kPostProcessThreshold;
    threshold(out, floodClone, floodThresh, 1.0f, THRESH_TOZERO);
    Mat mask = floodKillEdges(floodClone);
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
	return unscalePoint(maxP,eye);
}



void findEyes(Mat frame_gray, Rect face)
{
Mat faceROI = frame_gray(face);
Mat debugFace = faceROI;

//-- Find eye regions and draw them
int eye_region_width = face.width * (kEyePercentWidth/100.0); //ANTROPOMETRIC RATIOS
int eye_region_height = face.width * (kEyePercentHeight/100.0);
int eye_region_top = face.height * (kEyePercentTop/100.0);
Rect rightEyeRegion(face.width*(kEyePercentSide/100.0),eye_region_top,eye_region_width,eye_region_height);
Rect leftEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),eye_region_top,eye_region_width,eye_region_height);

//-- Find Eye Centers
leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");

wleftPupil=leftPupil;
wrightPupil=rightPupil;

// change eye centers to face coordinates
rightPupil.x += rightEyeRegion.x;
rightPupil.y += rightEyeRegion.y;
leftPupil.x += leftEyeRegion.x;
leftPupil.y += leftEyeRegion.y;

//change to moving face coordinate
rightPupil.x += face.x;
rightPupil.y += face.y;
leftPupil.x += face.x;
leftPupil.y += face.y;

rightEyeRegion.x += face.x;
rightEyeRegion.y += face.y;
leftEyeRegion.x += face.x;
leftEyeRegion.y += face.y;


// draw eye squares and the centers
rectangle(debugImage,leftEyeRegion,Scalar( 255, 0, 0 ));
rectangle(debugImage,rightEyeRegion,Scalar( 255, 0, 0 ));

//drawing the plus signs
Point l1,l2,l3,l4,l5,l6,l7,l8;

l1.x=rightPupil.x;
l2.x=rightPupil.x;
l1.y=(rightPupil.y-tada);
l2.y=(rightPupil.y+tada);

l3.y=rightPupil.y;
l4.y=rightPupil.y;
l3.x=(rightPupil.x-tada);
l4.x=(rightPupil.x+tada);

l5.x=leftPupil.x;
l6.x=leftPupil.x;
l5.y=(leftPupil.y-tada);
l6.y=(leftPupil.y+tada);

l7.y=leftPupil.y;
l8.y=leftPupil.y;
l7.x=(leftPupil.x-tada);
l8.x=(leftPupil.x+tada);

line(debugImage,l1,l2,Scalar(0,255,0),1,8);
line(debugImage,l3,l4,Scalar(0,255,0),1,8);
line(debugImage,l5,l6,Scalar(0,255,0),1,8);
line(debugImage,l7,l8,Scalar(0,255,0),1,8);

cv::Point lcol,rcol;
Point lefteyebrow,righteyebrow;

lcol.x=leftEyeRegion.x+(eye_region_width)/4-face.x;
lcol.y=leftEyeRegion.y+(eye_region_width)/3-face.y+2;

rcol.x=rightEyeRegion.x+3*(eye_region_width)/4-face.x;
rcol.y=rightEyeRegion.y+(eye_region_width)/3-face.y+2;

cv::Rect lcolm(lcol.x,lcol.y,2,(eye_region_width)/4),rcolm(rcol.x,rcol.y,2,(eye_region_width)/4);
//imshow("Left Eye",(faceROI(lcolm)));
//imshow("Right Eye",(faceROI(rcolm)));

cv::Mat lefthld,righthld;

equalizeHist( faceROI(lcolm), lefthld);
equalizeHist( faceROI(rcolm), righthld);

minMaxLoc(lefthld, NULL,NULL,&lefteyebrow,NULL);
minMaxLoc(righthld, NULL,NULL,&righteyebrow,NULL);

ylefteyebrow=lefteyebrow.y;
yrighteyebrow=righteyebrow.y;
//printf("%d \n ",(lefteyebrow.y+righteyebrow.y)/2);



lefteyebrow.x+=lcol.x+face.x;
lefteyebrow.y+=lcol.y+face.y;
righteyebrow.x+=rcol.x+face.x;
righteyebrow.y+=rcol.y+face.y;

circle(debugImage, righteyebrow, 2, cv::Scalar( 255, 255, 0 ),-1);
circle(debugImage, lefteyebrow, 2, cv::Scalar( 255, 255, 0 ),-1);

lcolm.x+=face.x;
lcolm.y+=face.y;
rcolm.x+=face.x;
rcolm.y+=face.y;
rectangle(debugImage,lcolm,cv::Scalar(0,255,255));
rectangle(debugImage,rcolm,cv::Scalar(0,255,255));


}




void guideit( Mat frame )
	{

	Mat frame_gray;
	std::vector<Rect> faces;
    //flip(frame, frame, 1);
	cvtColor( frame, frame_gray, CV_BGR2GRAY );

	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30,30) );   //150 150
      iboxx=oboxx+gap;
      iboxy=oboxy+gap;
      iboxw=oboxw-2*gap;
      iboxh=oboxh-2*gap;
    Rect obox(oboxx,oboxy,oboxw,oboxh);
	Rect ibox(iboxx,iboxy,iboxw,iboxh);

    rectangle(debugImage,ibox,Scalar( 255, 0, 0 ),3);
    rectangle(debugImage,obox,Scalar( 255, 0, 0 ),3);
        if(faces.size()>0)
        {
         //   faces[0].x=(faces[0].x/25)*25;
           // faces[0].y=(faces[0].y/25)*25;
            //faces[0].width=(faces[0].width/25)*25;
            //faces[0].height=(faces[0].height/25)*25;
            if(faces[0].x>oboxx && faces[0].y>oboxy && faces[0].width<oboxw && faces[0].height<oboxh)
            {
             if(faces[0].x<iboxx && faces[0].y<iboxy && faces[0].width>iboxw && faces[0].height>iboxh)
              {
                  poscor=1;
                  rectangle(debugImage,faces[0],Scalar(0,255,0),3);
                  findEyes(frame_gray, faces[0]);
              }
            else
                {
                    poscor=0;
                rectangle(debugImage,faces[0],Scalar(0,0,255),3);
                }
            }
            else
            {
                poscor=0;
                rectangle(debugImage,faces[0],Scalar(0,0,255),3);
            }
          //printf("%d %d %d %d \n ",faces[0].x,faces[0].y,faces[0].width,faces[0].height);
        }

	}
void caller(int,void*)
{
}

static void onMouse( int event, int x, int y, int, void* )
{
    if  ( event == CV_EVENT_LBUTTONDOWN  &&  poscor==1)
        {
         //fprintf(res,"%d %d %d %d %d %d \n",wleftPupil.x,wleftPupil.y,wrightPupil.x,wrightPupil.y,x,y);
        if(clkcnt<6)
            {
             fprintf(res,"%d %d\n",x,(wleftPupil.x+wrightPupil.x)/2);
             printf("%d %d\n",x,(wleftPupil.x+wrightPupil.x)/2);
             clkcnt++;
            }
        else if(clkcnt<12)
            {
             fprintf(res,"%d %d\n",y,(yrighteyebrow+ylefteyebrow)/2);
             printf("%d %d\n",y,(yrighteyebrow+ylefteyebrow)/2);
             clkcnt++;
            }
         //printf("%d %d %d %d %d %d \n",wleftPupil.x,wleftPupil.y,wrightPupil.x,wrightPupil.y,x,y);
        }
    else    if  ( event == CV_EVENT_RBUTTONDOWN  &&  poscor==1)
        {
         //fprintf(res,"%d %d %d %d %d %d \n",wleftPupil.x,wleftPupil.y,wrightPupil.x,wrightPupil.y,x,y);
            fprintf(res,"%d %d\n",(wleftPupil.x+wrightPupil.x)/2,x);
            printf("%d %d\n",(wleftPupil.x+wrightPupil.x)/2,x);
         //printf("%d %d %d %d %d %d \n",wleftPupil.x,wleftPupil.y,wrightPupil.x,wrightPupil.y,x,y);
        }
}
int main()
	{
	    CvCapture* capture;
        Mat frame;
        if( !face_cascade.load( face_cascade_name ) )
            {
            printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
            return -1;
            }
        res=fopen("data.txt","a");
       //setMouseCallback("face guide", onMouse, 0);

        namedWindow("face guide",CV_WINDOW_NORMAL);
        moveWindow("face guide", 400, 100);

        setMouseCallback("face guide", onMouse, 0);


        namedWindow("box size panel",CV_WINDOW_NORMAL);
        moveWindow("box size panel", 200, 100);

        createTrackbar( "boxTop","box size panel", &oboxy,500, caller);
        createTrackbar( "boxSide","box size panel", &oboxx,500, caller );
        createTrackbar( "boxWidth","box size panel", &oboxw,500, caller);
        createTrackbar( "boxHeight","box size panel", &oboxh,500, caller);
        createTrackbar( "gap between","box size panel", &gap,30, caller);

        capture = cvCaptureFromCAM( -1 );
        if( capture )
            {
            while( true )
                {
                frame = cvQueryFrame( capture );
                cv::flip(frame, frame, 1);
                frame.copyTo(debugImage);
                guideit(frame);
               /* Point strt,end;
                strt.x=213;
                strt.y=0;
                end.x=213;
                end.y=480;
                line(debugImage,strt,end,Scalar(255,0,255),2,8);
                strt.x=426;
                strt.y=0;
                end.x=426;
                end.y=480;
                line(debugImage,strt,end,Scalar(255,0,255),2,8);
                strt.x=0;
                strt.y=160;
                end.x=640;
                end.y=160;
                line(debugImage,strt,end,Scalar(255,0,255),2,8);
                 strt.x=0;
                strt.y=320;
                end.x=640;
                end.y=320;
                line(debugImage,strt,end,Scalar(255,0,255),2,8);*/
                Point clkpt;
                if(clkcnt<6)
                {
                clkpt.y=240;
                clkpt.x=53+clkcnt*106;
                circle(debugImage, clkpt, 10, Scalar( 255, 255, 255 ),-1,8);
                }
                else if(clkcnt<12)
                {
                    clkpt.x=320;
                    clkpt.y=40+(clkcnt-6)*80;
                    circle(debugImage, clkpt, 10, Scalar( 255, 255 ,255 ),-1,8);
                }


                imshow("face guide",debugImage);
                int c = waitKey(10);
                if( (char)c == 'c' ) { fclose(res);break; }
                }
            }
    return 0;
	}
