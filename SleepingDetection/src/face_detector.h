#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

class FaceDetector
{
public:
	FaceDetector();
    FaceDetector(const std::string cascadeFilePath);
    ~FaceDetector();

    cv::Point               Detect(cv::Mat frame);
    void                    setFaceCascade(const std::string cascadeFilePath);
    cv::CascadeClassifier*  faceCascade() const;
    void                    setResizedWidth(const int width);
    int                     resizedWidth() const;
	bool					isFaceFound() const;
    void                    resetFace();
    cv::Rect                face() const;
    cv::Point               facePosition() const;
    void                    setTemplateMatchingMaxDuration(const double s);
    double                  templateMatchingMaxDuration() const;
    double                  get_time(void) const;

private:
    static const double     TICK_FREQUENCY;

    dlib::frontal_face_detector m_frontal_face_detector;
    cv::CascadeClassifier*  m_faceCascade = NULL;
    std::vector<cv::Rect>   m_allFaces;
    cv::Rect                m_trackedFace;
    cv::Rect                m_faceRoi;
    cv::Mat                 m_faceTemplate;
    cv::Mat                 m_matchingResult;
    bool                    m_templateMatchingRunning = false;
    int64                   m_templateMatchingStartTime = 0;
    int64                   m_templateMatchingCurrentTime = 0;
    bool                    m_foundFace = false;
    double                  m_scale;
    int                     m_resizedWidth = 320;
    cv::Point               m_facePosition;
    cv::Point               m_facePositionPrevious;
    double                  m_templateMatchingMaxDuration = 3;

    cv::Rect    doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const;
    cv::Rect    biggestFace(std::vector<cv::Rect> &faces) const;
    cv::Point   centerOfRect(const cv::Rect &rect) const;
    cv::Mat     getFaceTemplate(const cv::Mat &frame, cv::Rect face);
    void        detectFaceAllSizes(const cv::Mat &frame);
    void        detectFaceAroundRoi(const cv::Mat &frame);
    void        detectFacesTemplateMatching(const cv::Mat &frame);
};

