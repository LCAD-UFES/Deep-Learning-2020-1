/*********************************************************
	---   Skeleton Module Application ---
 **********************************************************/


#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/shape_predictor.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "face_detector.h"
#include <cstring>
#include "caffednn.h"
#include <thread>
#include "argparse/argparse.hpp"
#include <dirent.h>
#include <chrono>         // std::chrono::seconds

#define DEBUG 0

using namespace std;
using namespace cv;


int ret_status = 0;

double IMAGE_WIDTH = -1;
double IMAGE_HEIGHT = -1;
double DESIRED_WIDTH_ROI = 256; // 256*3 = 768 (original desired roi width)
double DESIRED_HEIGHT_ROI = 192; // 192*3 = 576 (original desired roi height)
double roi_scale = 1; // between 0 and 1
double head_saving_image_time_interval = 5; // seconds

int FRAME_RATE = 15;

bool ALLOWED_TO_RESIZE = true;

static double camera_nominal_fps = 15.0;

typedef struct
{
    int left_eye_state;
    int right_eye_state;
}EyesEstimation;

Classifier *eyesDnnCaffe;
Classifier *earsDnnCaffe;

FaceDetector* detector;

string PREDICTOR_FILE_PATH;
string CASCADE_FILE_PATH;

dlib::shape_predictor predictor;

//3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
std::vector<cv::Point3d> object_pts;

//2D ref points(image coordinates), referenced from detected facial feature
std::vector<cv::Point2d> image_pts;

//reprojected 2D points
std::vector<cv::Point2d> reprojectdst;

//source 2D points
std::vector<cv::Point3d> reprojectsrc;

cv::Mat cam_matrix;
cv::Mat dist_coeffs;
cv::Mat gray_to_classify;

//result
cv::Mat rotation_vec;                           //3 x 1
cv::Mat rotation_mat;                           //3 x 3 R
cv::Mat translation_vec;                        //3 x 1 T
cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

//temp buf for decomposeProjectionMatrix()
cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

string VIDEOS_PATH = "";
vector<string> VIDEO_FILES;

string url = "";
int camera = 0;

string CAMERA = "";
string FILE_PATH = "";

std::string timeStampToHReadble(long long timestamp)
{
    const time_t rawtime = (const time_t)timestamp;
    struct tm * dt;
    char buffer [30];
    dt = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%Y%m%d/%H", dt);
    return std::string(buffer);
}

template<typename T>
void pop_front(std::vector<T> &v){
    if(v.size() > 0)
        v.erase(v.begin());
}

Point2d apply_homography( const Point2d& _point, const Mat& _H )
{
    Point2d ret = Point2d( -1, -1 );

    const double u = _H.at<double>(0,0) * _point.x + _H.at<double>(0,1) * _point.y + _H.at<double>(0,2);
    const double v = _H.at<double>(1,0) * _point.x + _H.at<double>(1,1) * _point.y + _H.at<double>(1,2);
    const double s = _H.at<double>(2,0) * _point.x + _H.at<double>(2,1) * _point.y + _H.at<double>(2,2);
    if ( s != 0 )
    {
        ret.x = ( u / s );
        ret.y = ( v / s );
    }
    return ret;
}

int dnnCheckEyesState(Mat candidate, Classifier *dnnCaffe) {

    int label = 1;
    double confidence;

    std::vector<Prediction>  predictions = dnnCaffe->Classify(candidate);
    Prediction p_1 = predictions[0];

    if ((!strcmp(p_1.first.c_str(),"na"))){
        label = 2;
    }else if ((!strcmp(p_1.first.c_str(),"closed"))){
        label = 0;
    }else if((!strcmp(p_1.first.c_str(),"open"))){
        label = 1;
    }else{ // forced
        Prediction p_2 = predictions[1];
        if ((!strcmp(p_2.first.c_str(),"closed"))){
            label = 0;
        }else{
            label = 1;
        }
    }

    return label;
}

cv::Rect restrict_roi(cv::Mat frame, cv::Rect roi)
{
    cv::Rect new_roi;
    new_roi = roi;

    if(new_roi.x < 0)  new_roi.x = 0;
    if(new_roi.y < 0)  new_roi.y = 0;
    if((new_roi.x + new_roi.width) >= frame.cols) new_roi.width = frame.cols - new_roi.x;
    if((new_roi.y + new_roi.height) >= frame.rows) new_roi.height = frame.rows - new_roi.y;

    return new_roi;
}

void DrawHead(Mat& frame, cv::Rect face_rect, dlib::full_object_detection shape, EyesEstimation eyes_state, int sleeping_state, Mat homography)
{
    std::ostringstream outtext;
    cv::putText(frame, outtext.str(), cv::Point(frame.cols - 100, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255));
    outtext.str("");

    char left_eye_text[256];
    char right_eye_text[256];
    char text[256];

    //cv::rectangle(frame, face_rect, cv::Scalar(0, 255, 255), 1);
    //cv::line(frame, cv::Point2i(0, 100), cv::Point2i(frame.cols - 1, 100), Scalar(0, 255, 255), 1);

    for (unsigned int i = 0; i < 68; ++i)
        cv::circle(frame, cv::Point(shape.part(i).x(), shape.part(i).y()), 1, cv::Scalar(255, 255, 255), -1);

    //draw axis
    cv::line(frame, reprojectdst[0], reprojectdst[1], cv::Scalar(0, 0, 255)); //top_left
    cv::line(frame, reprojectdst[1], reprojectdst[2], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[2], reprojectdst[3], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[3], reprojectdst[0], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[4], reprojectdst[5], cv::Scalar(0, 0, 255)); //top_right
    cv::line(frame, reprojectdst[5], reprojectdst[6], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[6], reprojectdst[7], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[7], reprojectdst[4], cv::Scalar(0, 0, 255)); //bottom_right
    cv::line(frame, reprojectdst[0], reprojectdst[4], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[1], reprojectdst[5], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[2], reprojectdst[6], cv::Scalar(0, 0, 255));
    cv::line(frame, reprojectdst[3], reprojectdst[7], cv::Scalar(0, 0, 255)); //bottom_left

    cv::Mat label_estimations;
    frame.copyTo(label_estimations);

    cv::Point  pt2[1][4];
    pt2[0][0] = cv::Point2d(0.35 * frame.cols, frame.rows - 105);
    pt2[0][1] = cv::Point2d(0.85 * frame.cols, frame.rows - 105);
    pt2[0][2] = cv::Point2d(0.85 * frame.cols, frame.rows - 10);
    pt2[0][3] = cv::Point2d(0.35 * frame.cols, frame.rows - 10);

    const cv::Point * ppt2[1] = {pt2[0]};
    int npt2[] = {4};
    cv::fillPoly(label_estimations, ppt2, npt2, 1, cv::Scalar(255, 255, 255));

    cv::addWeighted(label_estimations, 0.8, frame, 0.2, 0, frame);

    // eye state 0 = closed
    // eye state 1 = open
    // eye state 2 = na

    if(eyes_state.left_eye_state == 0)
        sprintf(left_eye_text, "Closed");
    else if(eyes_state.left_eye_state == 1)
        sprintf(left_eye_text, "Open");
    else if(eyes_state.left_eye_state == 2)
        sprintf(left_eye_text, "Na");

    else
        sprintf(left_eye_text, "No Detection");

    if(eyes_state.right_eye_state == 0)
        sprintf(right_eye_text, "Closed");
    else if(eyes_state.right_eye_state == 1)
        sprintf(right_eye_text, "Open");
    else if(eyes_state.right_eye_state == 2)
        sprintf(right_eye_text, "Na");
    else
        sprintf(right_eye_text, "No Detection");

    sprintf(text, "Left Eye: %s", left_eye_text);
    cv::putText(frame, text, cv::Point(frame.cols*0.42, frame.rows - 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0), 2);
    sprintf(text, "Right Eye: %s", right_eye_text);
    cv::putText(frame, text, cv::Point(frame.cols*0.42, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0), 2);

    // sleeping state 0 = awake
    // sleeping state 1 = sleeping
    // sleeping state 2 = no detection

    if(sleeping_state == 0)
        sprintf(text, "%s", "AWAKE");
    else if(sleeping_state == 1)
        sprintf(text, "%s", "SLEEPING");
    else if(sleeping_state == 2)
        sprintf(text, "%s", "NO DETECTION");

    cv::putText(frame, text, cv::Point(frame.cols*0.46, frame.rows - 80), cv::FONT_HERSHEY_DUPLEX, 0.8, CV_RGB(0, 58, 89), 2);
}

cv::Mat warp_face(cv::Mat frame, dlib::full_object_detection shape, std::vector<cv::Point2d> reprojectdst, std::vector<cv::Point2d>& shape_warped, cv::Mat& homography)
{
    // Applying perspective:
    cv::Mat warp_img;
    warp_img = cv::Mat::zeros(200, 200, CV_8UC3);

    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> destPoints;

    srcPoints.push_back(reprojectdst[0]);
    srcPoints.push_back(reprojectdst[4]);
    srcPoints.push_back(reprojectdst[7]);
    srcPoints.push_back(reprojectdst[3]);

    destPoints.push_back(cv::Point2f(0, 0));
    destPoints.push_back(cv::Point2f(warp_img.cols, 0));
    destPoints.push_back(cv::Point2f(warp_img.cols, warp_img.rows));
    destPoints.push_back(cv::Point2f(0, warp_img.rows));

    cv::Mat warpMatrix = cv::findHomography(srcPoints,destPoints);

    cv::warpPerspective(frame, warp_img, warpMatrix,warp_img.size());


    for(int i = 0; i < shape.num_parts(); i++)
    {
        cv::Point2d warped_point = apply_homography(cv::Point2d(shape.part(i).x(), shape.part(i).y()), warpMatrix);
        shape_warped.push_back(warped_point);
    }

    homography = warpMatrix.clone();

    imshow("warp_img", warp_img);

    return warp_img;
}

Rect2d crop_eyes_with_eyebrows(Mat frame, Point2f left_point_eye, Point2f right_point_eye, Point2f left_eyebrow, Point2f top_eyebrow, Point2f right_eyebrow)
{
    double eye_y_dist = 2*fabs(left_point_eye.x - right_point_eye.x);
    double eye_up_limit_height = 2*fabs(left_point_eye.x - right_point_eye.x);
    if(eye_y_dist > eye_up_limit_height) eye_y_dist = eye_up_limit_height;

    double tl_x = left_eyebrow.x;
    double tl_y = top_eyebrow.y - eye_y_dist*0.15;
    double br_x = right_eyebrow.x;
    double br_y = top_eyebrow.y + eye_y_dist*0.85;

    if(tl_x < 0) tl_x = 0;
    if(tl_y < 0) tl_y = 0;
    if(br_x > frame.cols) br_x = frame.cols;
    if(br_y > frame.rows) br_y = frame.rows;

    cv::Point2d roi_top_left_corner = cv::Point2d(tl_x, tl_y);
    cv::Point2d roi_bottom_right_corner = cv::Point2d(br_x, br_y);

    cv::Rect2d rect_roi(roi_top_left_corner,roi_bottom_right_corner);
    rect_roi = restrict_roi(frame, rect_roi);

    return rect_roi;
}

cv::Mat crop_roi_in_face(cv::Mat& frame, int top_left_x, int top_left_y, int bottom_right_x, int bottom_right_y){

    double tl_x = top_left_x;
    double tl_y = top_left_y;
    double br_x = bottom_right_x;
    double br_y = bottom_right_y;

    if(tl_x < 0) tl_x = 0;
    if(tl_y < 0) tl_y = 0;
    if(br_x > frame.cols) br_x = frame.cols;
    if(br_y > frame.rows) br_y = frame.rows;

    cv::Point2d roi_top_left_corner = cv::Point2d(tl_x, tl_y);
    cv::Point2d roi_bottom_right_corner = cv::Point2d(br_x, br_y);

    cv::Rect2d rect_roi(roi_top_left_corner,roi_bottom_right_corner);

    rect_roi = restrict_roi(frame, rect_roi);

    return frame(rect_roi).clone();
}

int get_max_index_value(vector<int> v)
{
    int count_max = -9999;
    int count_max_index = 0;

    for(int i = 0; i < v.size(); ++i)
    {   
        if(v[i] > count_max)
        {
            count_max = v[i];
            count_max_index = i;
        }
    }

    return count_max_index;
}

void increment_state(int* count, int memory)
{
    (*count)++;
    if (*count >= memory)
        *count = memory;
}

void decrement_state(int* count, int memory)
{
    (*count)--;
    if (*count <= -memory)
        *count = -memory;
}

int get_eyes_estimation(cv::Mat& frame, std::vector<cv::Point2d> shape, Classifier *dnnCaffe, EyesEstimation& eyes_state)
{
    static int left_eye_closed_state_count = 0;
    static int left_eye_open_state_count = 0;
    static int left_eye_na_state_count = 0;
    static int right_eye_closed_state_count = 0;
    static int right_eye_open_state_count = 0;
    static int right_eye_na_state_count = 0;

    const int EYE_STATE_MEMORY = (camera_nominal_fps / 3);

    // Left eye
    Rect2d left_eye_rect = crop_eyes_with_eyebrows(frame, shape[42], shape[45], shape[22], shape[24],
                                                    shape[26]);
    Mat left_eye_roi = frame(left_eye_rect).clone();
    if ((left_eye_roi.cols * left_eye_roi.rows) > 0) {
        Mat left_eye, left_eye_clahe;
        resize(left_eye_roi, left_eye, Size(32, 32), CV_INTER_LINEAR);
        cv::flip(left_eye, left_eye, 1);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2);
        clahe->setTilesGridSize(Size(2, 2));
        clahe->apply(left_eye, left_eye_clahe);
        normalize(left_eye_clahe, left_eye_clahe, 0, 255, NORM_MINMAX);

        imshow("left_eye", left_eye_clahe);
    
        if (dnnCheckEyesState(left_eye_clahe, dnnCaffe) == 0) // closed
        {
            increment_state(&left_eye_closed_state_count, EYE_STATE_MEMORY);
            decrement_state(&left_eye_open_state_count, EYE_STATE_MEMORY);
            decrement_state(&left_eye_na_state_count, EYE_STATE_MEMORY);

        }else if(dnnCheckEyesState(left_eye_clahe, dnnCaffe) == 1) // open
        {
            decrement_state(&left_eye_closed_state_count, EYE_STATE_MEMORY);
            increment_state(&left_eye_open_state_count, EYE_STATE_MEMORY);
            decrement_state(&left_eye_na_state_count, EYE_STATE_MEMORY);
        }else // na
        {
            decrement_state(&left_eye_closed_state_count, EYE_STATE_MEMORY);
            decrement_state(&left_eye_open_state_count, EYE_STATE_MEMORY);
            increment_state(&left_eye_na_state_count, EYE_STATE_MEMORY);
        }
    }
    
    // Right eye
    Rect2d right_eye_rect = crop_eyes_with_eyebrows(frame, shape[36], shape[39], shape[17], shape[19], shape[21]);
    Mat right_eye_roi = frame(right_eye_rect).clone();
    if((right_eye_roi.cols * right_eye_roi.rows) > 0)
    {
        Mat right_eye, right_eye_clahe;
        resize(right_eye_roi, right_eye, Size(32, 32), CV_INTER_LINEAR);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2);
        clahe->setTilesGridSize(Size(2, 2));
        clahe->apply(right_eye, right_eye_clahe);
        normalize(right_eye_clahe, right_eye_clahe, 0, 255, NORM_MINMAX);

        imshow("right_eye", right_eye_clahe);

        if (dnnCheckEyesState(right_eye_clahe, dnnCaffe) == 0) // closed
        {
            increment_state(&right_eye_closed_state_count, EYE_STATE_MEMORY);
            decrement_state(&right_eye_open_state_count, EYE_STATE_MEMORY);
            decrement_state(&right_eye_na_state_count, EYE_STATE_MEMORY);

        }else if(dnnCheckEyesState(right_eye_clahe, dnnCaffe) == 1) // open
        {
            decrement_state(&right_eye_closed_state_count, EYE_STATE_MEMORY);
            increment_state(&right_eye_open_state_count, EYE_STATE_MEMORY);
            decrement_state(&right_eye_na_state_count, EYE_STATE_MEMORY);
        }else // na
        {
            decrement_state(&right_eye_closed_state_count, EYE_STATE_MEMORY);
            decrement_state(&right_eye_open_state_count, EYE_STATE_MEMORY);
            increment_state(&right_eye_na_state_count, EYE_STATE_MEMORY);
        }
    }

    vector<int> left_eye_states_count = {left_eye_closed_state_count, left_eye_open_state_count, left_eye_na_state_count};
    eyes_state.left_eye_state = get_max_index_value(left_eye_states_count);

    vector<int> right_eye_states_count = {right_eye_closed_state_count, right_eye_open_state_count, right_eye_na_state_count};
    eyes_state.right_eye_state = get_max_index_value(right_eye_states_count);

    return 0;
}

int get_sleeping_estimation(EyesEstimation eyes_state, int *sleeping_state)
{
    // eye state 0 = closed
    // eye state 1 = open
    // eye state 2 = na
    // sleeping state 0 = awake
    // sleeping state 1 = sleeping
    // sleeping state 2 = no detection

    static bool eyes_were_not_closed = true;
    
    if(eyes_state.left_eye_state == 0 && eyes_state.right_eye_state == 0) // eyes closed
    {
        static double start_time = detector->get_time();

        if(eyes_were_not_closed){
            start_time = detector->get_time();
            eyes_were_not_closed = false;
        }
            
        double end_time = detector->get_time();
        
        if(fabs(end_time-start_time) > 3.0){
            *sleeping_state = 1;
        }
    }else if(eyes_state.left_eye_state == 1 && eyes_state.right_eye_state == 1) // eyes open
    {   
        *sleeping_state = 0;
        eyes_were_not_closed = true;
    }else if(eyes_state.left_eye_state == 2 || eyes_state.right_eye_state == 2) // eyes na
    {   
        *sleeping_state = 2;
        eyes_were_not_closed = true;
    }

    return 0;
}

void initialize_head_detection(int image_width, int image_height)
{
    detector = new FaceDetector(CASCADE_FILE_PATH);
    detector->setResizedWidth(280);
    detector->setTemplateMatchingMaxDuration(1.0);

    dlib::deserialize(PREDICTOR_FILE_PATH) >> predictor;

    //fill in cam intrinsics and distortion coefficients
    double K[9] = { (double) image_width, 0.0, ((double) image_width / 2.0), 0.0, (double) image_width, ((double) image_height / 2.0), 0.0, 0.0, 1.0 };
    double D[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    cv::Mat(3, 3, CV_64FC1, K).copyTo(cam_matrix);
    cv::Mat(5, 1, CV_64FC1, D).copyTo(dist_coeffs);

    //fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    object_pts.push_back(cv::Point3d(6.825897,6.760612,4.402142));     //#33 left brow left corner
    object_pts.push_back(cv::Point3d(1.330353,7.122144,6.903745));     //#29 left brow right corner
    object_pts.push_back(cv::Point3d(-1.330353,7.122144,6.903745));    //#34 right brow left corner
    object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner

//    object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
//    object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
//    object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
//    object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

    //reproject 3D points world coordinate axis to verify result pose
    reprojectsrc.push_back(cv::Point3d(8.0, 12.0, 8.0));
    reprojectsrc.push_back(cv::Point3d(8.0, 12.0, -8.0));
    reprojectsrc.push_back(cv::Point3d(8.0, -8.0, -8.0));
    reprojectsrc.push_back(cv::Point3d(8.0, -8.0, 8.0));
    reprojectsrc.push_back(cv::Point3d(-8.0, 12.0, 8.0));
    reprojectsrc.push_back(cv::Point3d(-8.0, 12.0, -8.0));
    reprojectsrc.push_back(cv::Point3d(-8.0, -8.0, -8.0));
    reprojectsrc.push_back(cv::Point3d(-8.0, -8.0, 8.0));

    //reprojected 2D points
    reprojectdst.resize(8);
}

void open_video(VideoCapture* cap)
{
    if(FILE_PATH != "")
    {
        cap->open(FILE_PATH);
        FRAME_RATE = cap->get(CV_CAP_PROP_FPS);
    }else{
        cap->open(stoi(CAMERA));
        FRAME_RATE = 30;
        cap->set(CV_CAP_PROP_FPS, FRAME_RATE);
    }

    if(!cap->isOpened()){
        cout << "Error opening video stream or file" << endl;
        exit(1);
    }
}

void detect_sleeping()
{
    static int first_time = 1;
    int sleeping_state = 0;
    int face_is_detected = 0;

    EyesEstimation eyes_state;
    eyes_state.left_eye_state = 1; // open
    eyes_state.right_eye_state = 1; // open

    VideoCapture cap;
    open_video(&cap);

    if(IMAGE_WIDTH == -1 || IMAGE_HEIGHT == -1)
    {
        IMAGE_WIDTH = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        IMAGE_HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        ALLOWED_TO_RESIZE = false;
    }

    initialize_head_detection(IMAGE_WIDTH, IMAGE_HEIGHT);

    while(true)
    {  
        auto st_time = std::chrono::high_resolution_clock::now();
        Mat gray;
        std::vector<dlib::rectangle> faces;

        Mat frame;
        cap >> frame;

        if((frame.empty())){
            break;
        }

        if(ALLOWED_TO_RESIZE)
            resize(frame, frame, Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_INTER_LINEAR);

        cv::cvtColor(frame, gray_to_classify, CV_BGR2GRAY);

        // Applying CLAHE
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(1);
        clahe->setTilesGridSize(Size(3, 3));
        clahe->apply(gray, gray);

        cv::Mat mini_gray;
        dlib::cv_image<unsigned char> cimg(gray);

        // Detect faces
        detector->Detect(gray);

        // Find the pose of each face
        if (detector->isFaceFound())
        {
            cv::Rect face_rect = detector->face();

            if(face_rect.x != 0 && face_rect.y != 0 && face_rect.x != (gray.cols - 1) && face_rect.y != (gray.rows - 1))
            {
                face_is_detected = 1;

                dlib::rectangle face(face_rect.x, face_rect.y, face_rect.x + face_rect.width, face_rect.y + face_rect.height);

                //track features
                dlib::full_object_detection shape = predictor(cimg, face);

                //fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
                image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
                image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
                image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
                image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
                image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
                image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
                image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
                image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
                image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
                image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
                //                image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
                //                image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
                //                image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
                //                image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

                //calc pose to get the rotation and translation matrix
                cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

                //reproject (red cube)
                cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);
                
                // initializing shape_warped
                cv::Mat homography;
                std::vector<cv::Point2d> shape_warped;
                cv::Mat face_warped = warp_face(gray_to_classify, shape, reprojectdst, shape_warped, homography);

                get_eyes_estimation(face_warped, shape_warped, eyesDnnCaffe, eyes_state);

                get_sleeping_estimation(eyes_state, &sleeping_state);

                DrawHead(frame, face_rect, shape, eyes_state, sleeping_state, homography);

                // research face in whole image in case of losing track
                static int face_reset_count = 0;
                if(eyes_state.left_eye_state == 2 && eyes_state.right_eye_state == 2){
                    detector->resetFace();
                    image_pts.clear();
                    face_reset_count = 0;
                    continue;
                }

                face_reset_count++;
                image_pts.clear();
            }
        }
    
        cv::imshow("detection", frame);
        cv::waitKey(1);  

        auto ed_time = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time-st_time);
        auto duration = std::chrono::nanoseconds((int)(1000000000.0 / FRAME_RATE));
        std::this_thread::sleep_for(duration - delta);      
    }
}

std::string get_dnn_path() 
{
    system("cd ../net_models && pwd > ../build/temp.txt");
    std::ifstream ifs("temp.txt");
    std::string ret{ std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>() };
    ifs.close(); // must close the inout stream so the file can be cleaned up
    system("cd ../build");
    if (std::remove("temp.txt") != 0) {
        perror("Error deleting temporary file");
    }
    ret[ret.size()-1] = '/';
    return ret;
}

void load_dnn_classifiers()
{
    string DNN_PATH = get_dnn_path();

    // loading shape predictor:
    PREDICTOR_FILE_PATH = DNN_PATH + "faces/shape_predictor_68_face_landmarks.dat";

    // loading haar cascade:
    CASCADE_FILE_PATH = DNN_PATH + "faces/haarcascade_frontalface_default.xml";

    // loading eyes network:
    string EYES_MODEL_FILE_PATH = DNN_PATH + "eyes/deploy.prototxt";
    string EYES_TRAINED_FILE_PATH = DNN_PATH + "eyes/snapshot.caffemodel";
    string EYES_MEAN_FILE_PATH = DNN_PATH + "eyes/mean.binaryproto";
    string EYES_LABEL_FILE_PATH = DNN_PATH + "eyes/labels.txt";
    eyesDnnCaffe = new Classifier(EYES_MODEL_FILE_PATH, EYES_TRAINED_FILE_PATH, EYES_MEAN_FILE_PATH, EYES_LABEL_FILE_PATH);
}

bool is_a_video(char *name)
{
    if (name == NULL)
        return false;

    if (strcasestr(name, ".aaf") != NULL ||
        strcasestr(name, ".3gp") != NULL ||
        strcasestr(name, ".gif") != NULL ||
        strcasestr(name, ".asf") != NULL ||
        strcasestr(name, ".wma") != NULL ||
        strcasestr(name, ".wmv") != NULL ||
        strcasestr(name, ".m2ts") != NULL ||
        strcasestr(name, ".mts") != NULL ||
        strcasestr(name, ".avi") != NULL ||
        strcasestr(name, ".cam") != NULL ||
        strcasestr(name, ".dat") != NULL ||
        strcasestr(name, ".dsh") != NULL ||
        strcasestr(name, ".dvr-ms") != NULL ||
        strcasestr(name, ".flv") != NULL ||
        strcasestr(name, ".f4v") != NULL ||
        strcasestr(name, ".f4p") != NULL ||
        strcasestr(name, ".f4a") != NULL ||
        strcasestr(name, ".f4b") != NULL ||
        strcasestr(name, ".mpg") != NULL ||
        strcasestr(name, ".mpeg") != NULL ||
        strcasestr(name, ".m1v") != NULL ||
        strcasestr(name, ".mpv") != NULL ||
        strcasestr(name, ".fla") != NULL ||
        strcasestr(name, ".flr") != NULL ||
        strcasestr(name, ".sol") != NULL ||
        strcasestr(name, ".m4v") != NULL ||
        strcasestr(name, ".mkv") != NULL ||
        strcasestr(name, ".wrap") != NULL ||
        strcasestr(name, ".mng") != NULL ||
        strcasestr(name, ".mov") != NULL ||
        strcasestr(name, ".mp4") != NULL ||
        strcasestr(name, ".mpe") != NULL ||
        strcasestr(name, ".mxf") != NULL ||
        strcasestr(name, ".roq") != NULL ||
        strcasestr(name, ".nsv") != NULL ||
        strcasestr(name, ".ogg") != NULL ||
        strcasestr(name, ".rm") != NULL ||
        strcasestr(name, ".svi") != NULL ||
        strcasestr(name, ".wmv") != NULL ||
        strcasestr(name, ".wtv") != NULL)
    {
        return true;
    }

    return false;
}

vector<string> read_directory(string path)
{
    if (path[path.size()-1] != '/')
        path += '/';
    dirent *de;
    DIR *dp;
    errno = 0;
    dp = opendir(path.empty() ? "." : path.c_str());
    vector<string> video_files;
    if (dp)
    {
        while (true)
        {
            errno = 0;
            de = readdir(dp);
            if (de == NULL)
                break;
            if (is_a_video(de->d_name))
                video_files.push_back(std::string(path+de->d_name));
        }
        closedir(dp);
        std::sort(video_files.begin(), video_files.end());
    }
    return video_files;
}

void read_parameters(int argc, char** argv)
{
    ArgumentParser parser;
    parser.addArgument("-c", "--camera", 1);
    parser.addArgument("-f", "--file", 1);
    parser.addArgument("-p", "--path", 1);
    parser.addArgument("-w", "--width", 1);
    parser.addArgument("-h", "--height", 1);
    parser.parse(argc, (const char **)argv);

    float width = -1;
    float height = -1;

    if (parser.exists("width"))
        IMAGE_WIDTH = parser.retrieve<int>("width");

    if (parser.exists("height"))
        IMAGE_HEIGHT = parser.retrieve<int>("height");

    if (parser.exists("camera"))
        CAMERA = parser.retrieve("camera");

    if (parser.exists("file"))
        FILE_PATH = parser.retrieve("file");

    if (parser.exists("path"))
    {
        VIDEOS_PATH = parser.retrieve("path");
        VIDEO_FILES = read_directory(VIDEOS_PATH);

        for(int i = 0; i < VIDEO_FILES.size(); i++)
        {
            FILE_PATH = VIDEO_FILES[i];
            detect_sleeping();
        }
        return;
    }

    detect_sleeping();
}

int
main(int argc, char **argv)
{
    load_dnn_classifiers();

    read_parameters(argc, argv);

    return (0);
}
