#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "seeta/face_detection.h"
#include "seeta/face_alignment.h"
#include "seeta/face_identification.h"
#include "alize/alize.h"

float FaceCompare(const std::string path_src, const std::string path_dst) {

    seeta::FaceDetection faceDetection("../models/seeta_fd_frontal_v1.0.bin");
    seeta::FaceAlignment faceAlignment("../models/seeta_fa_v1.1.bin");
    seeta::FaceIdentification faceIdentification("../models/seeta_fr_v1.0.bin");

    cv::Mat mat_src = cv::imread(path_src);
    cv::Mat mat_dst = cv::imread(path_dst);

    cv::Mat mat_src_gray(mat_src.rows, mat_src.cols, CV_8UC1);
    cv::Mat mat_dst_gray(mat_dst.rows, mat_dst.cols, CV_8UC1);

    if (mat_src.empty() || mat_dst.empty()) {
        std::cout << "Failed to read images." << std::endl;
        return -1;
    }

    cv::cvtColor(mat_src, mat_src_gray, CV_BGR2GRAY);
    cv::cvtColor(mat_dst, mat_dst_gray, CV_BGR2GRAY);

    seeta::ImageData img_data_src_gray(mat_src_gray.cols, mat_src_gray.rows, 1);
    seeta::ImageData img_data_dst_gray(mat_dst_gray.cols, mat_dst_gray.rows, 1);
    img_data_src_gray.data = mat_src_gray.data;
    img_data_dst_gray.data = mat_dst_gray.data;

    std::vector<seeta::FaceInfo> faces_src = faceDetection.Detect(img_data_src_gray);
    std::vector<seeta::FaceInfo> faces_dst = faceDetection.Detect(img_data_dst_gray);
    if (faces_src.empty() || faces_dst.empty()) {
        std::cout << "Detected no faces." << std::endl;
        return -1;
    }

    std::vector<seeta::FacialLandmark> landmard_src(5);
    std::vector<seeta::FacialLandmark> landmard_dst(5);
    faceAlignment.PointDetectLandmarks(img_data_src_gray, faces_src[0], &landmard_src[0]);
    faceAlignment.PointDetectLandmarks(img_data_dst_gray, faces_dst[0], &landmard_dst[0]);

    seeta::ImageData img_data_src(mat_src.cols, mat_src.rows, mat_src.channels());
    seeta::ImageData img_data_dst(mat_dst.cols, mat_dst.rows, mat_dst.channels());
    img_data_src.data = mat_src.data;
    img_data_dst.data = mat_dst.data;
    cv::Mat mat_src_crop(faceIdentification.crop_height(), faceIdentification.crop_width(),
                         CV_8UC(faceIdentification.crop_channels()));
    cv::Mat mat_dst_crop(faceIdentification.crop_height(), faceIdentification.crop_width(),
                         CV_8UC(faceIdentification.crop_channels()));
    seeta::ImageData img_data_src_crop(mat_src_crop.cols, mat_src_crop.rows, mat_src_crop.channels());
    seeta::ImageData img_data_dst_crop(mat_dst_crop.cols, mat_dst_crop.rows, mat_dst_crop.channels());
    img_data_src_crop.data = mat_src_crop.data;
    img_data_dst_crop.data = mat_dst_crop.data;

    faceIdentification.CropFace(img_data_src, &landmard_src[0], img_data_src_crop);
    faceIdentification.CropFace(img_data_dst, &landmard_dst[0], img_data_dst_crop);

    int feat_size = faceIdentification.feature_size();
    float *feats_src = new float[feat_size];
    float *feats_dst = new float[feat_size];
    faceIdentification.ExtractFeature(img_data_src, feats_src);
    faceIdentification.ExtractFeature(img_data_dst, feats_dst);

    float sim = faceIdentification.CalcSimilarity(feats_src, feats_dst);

    return sim;

}


int main(int argc, char *argv[]) {

    string hint = "Compute similarity of two faces.(returns similarity between(0-1))\n";
    hint += "Usage:\n";
    hint += "    fr <path-to-face-jpg-src> <path-to-face-jpg-dst>\n";

    if (argc != 3) {
        std::cout << hint;
        return 0;
    }
    string path_src = argv[1];
    string path_dst = argv[2];
    std::cout << FaceCompare(path_src,path_dst) << std::endl;
    return 0;
}



