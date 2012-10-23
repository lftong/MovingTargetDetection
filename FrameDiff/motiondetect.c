#include "cv.h"
#include "highgui.h"
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

// various tracking parameters (in seconds)
const double MHI_DURATION = 0.5;//0.5sΪ�˶����ٵ�������ʱ��
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
const int N = 3;

//
const int CONTOUR_MAX_AERA = 16;

// ring image buffer	Ȧ��ͼ�񻺳�
IplImage **buf = 0;		//ָ���ָ��
int last = 0;

// temporary images
IplImage *mhi = 0; // MHI: motion history image

CvFilter filter = CV_GAUSSIAN_5x5;
CvConnectedComp *cur_comp, min_comp;
CvConnectedComp comp;
CvMemStorage *storage;
CvPoint pt[4];

//  ������
//  img �C ������Ƶ֡
//  dst �C �����
void  update_mhi( IplImage* img, IplImage* dst, int diff_threshold )
{
    double timestamp = clock()/100.; // get current time in seconds
    CvSize size = cvSize(img->width,img->height); // get current frame size
    int i, j, idx1, idx2;
    IplImage* silh;
    uchar val;
    float temp;
    IplImage* pyr = cvCreateImage( cvSize((size.width & -2)/2, (size.height & -2)/2), 8, 1 );
    CvMemStorage *stor;
    CvSeq *cont, *result, *squares;
    CvSeqReader reader;

    if( !mhi || mhi->width != size.width || mhi->height != size.height ) 
    {
        if( buf == 0 ) 
        {
            buf = (IplImage**)malloc(N*sizeof(buf[0]));
            memset( buf, 0, N*sizeof(buf[0]));
        }
        
        for( i = 0; i < N; i++ ) 
        {
            cvReleaseImage( &buf[i] );
            buf[i] = cvCreateImage( size, IPL_DEPTH_8U, 1 );
            cvZero( buf[i] );
        }
        cvReleaseImage( &mhi );
        mhi = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        cvZero( mhi ); // clear MHI at the beginning
    } // end of if(mhi)

    cvCvtColor( img, buf[last], CV_BGR2GRAY ); // convert frame to grayscale

    idx1 = last;
    idx2 = (last + 1) % N; // index of (last - (N-1))th frame 
    last = idx2;

    // ��֡��
    silh = buf[idx2];
    cvAbsDiff( buf[idx1], buf[idx2], silh ); // get difference between frames
    
    // �Բ�ͼ������ֵ��
    cvThreshold( silh, silh, 30, 255, CV_THRESH_BINARY ); // and threshold it
    
    cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // update MHI
    cvCvtScale( mhi, dst, 255./MHI_DURATION, 
      (MHI_DURATION - timestamp)*255./MHI_DURATION );    
    cvCvtScale( mhi, dst, 255./MHI_DURATION, 0 );    
    
    // ��ֵ�˲�������С������
    cvSmooth( dst, dst, CV_MEDIAN, 3, 0, 0, 0 );
    
    // ���²�����ȥ������
    cvPyrDown( dst, pyr, 7 );
    cvDilate( pyr, pyr, 0, 1 );  // �����Ͳ���������Ŀ��Ĳ������ն�
    cvPyrUp( pyr, dst, 7 );
    //
    // ����ĳ���������ҵ�����
    //
    // Create dynamic structure and sequence.
    stor = cvCreateMemStorage(0);
    cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint) , stor);
    
    // �ҵ���������
    cvFindContours( dst, stor, &cont, sizeof(CvContour), 
                    CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
/*
    for(;cont;cont = cont->h_next)
    {   
        // Number point must be more than or equal to 6 (for cvFitEllipse_32f).        
        if( cont->total < 6 )  
            continue;

        // Draw current contour.
        cvDrawContours(img,cont,CV_RGB(255,0,0),CV_RGB(255,0,0),0,1, 8, cvPoint(0,0));
    }  // end of for-loop: "cont"
*/
    // ֱ��ʹ��CONTOUR�еľ�����������
    for(;cont;cont = cont->h_next)
    {
              CvRect r = ((CvContour*)cont)->rect;
              if(r.height * r.width > CONTOUR_MAX_AERA) // ���С�ķ���������
              {
                  cvRectangle( img, cvPoint(r.x,r.y), 
                          cvPoint(r.x + r.width, r.y + r.height),
                          CV_RGB(255,0,0), 1, CV_AA,0);
              }
    }
    // free memory
    cvReleaseMemStorage(&stor);
    cvReleaseImage( &pyr );
}

int main(int argc, char** argv)
{
    IplImage* motion = 0; 
    CvCapture* capture = 0;
    
    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );
    else if( argc == 2 )
        capture = cvCaptureFromFile( argv[1] ); 
    if( capture )
    {
        cvNamedWindow( "Motion", 1 );
        for(;;)
        {
            IplImage* image;
            if( !cvGrabFrame( capture ))
                break;
            image = cvRetrieveFrame( capture , 0);
            if( image )
            {
                if( !motion )
                {
                    motion = cvCreateImage( cvSize(image->width,image->height), 8, 1 );
                    cvZero( motion );
                    motion->origin = image->origin;
                }
            }

            update_mhi( image, motion, 60 );
            cvShowImage( "Motion", image );

            if( cvWaitKey(10) >= 0 )
                break;
        }
        cvReleaseCapture( &capture );
        cvDestroyWindow( "Motion" );
    }
    return 0;
}
