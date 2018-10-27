
#include <iostream>
#include <string>
#include <cmath>
#include <set>

#include "opencv2/opencv.hpp"
#include "hough_trans.h"

HoughTrans::HoughTrans() {

}

HoughTrans::~HoughTrans() {

}

void HoughTrans::initialize(const HoughTransParams &params) {

  rho_theta_rows = int(params.rho_max / params.rho_res);
  rho_theta_cols = int(params.theta_max / params.theta_res);
  rho_max = params.rho_max;
  rho_res = params.rho_res;
  theta_res = params.theta_res;
  vote_thresh = params.vote_thresh;
  score_thresh = params.score_thresh;
  rho_theta = cv::Mat::zeros(rho_theta_rows, rho_theta_cols, CV_32S);

}

void HoughTrans::get_top_ranked_lines(std::vector<LineElem> &lines) {

  lines.clear();

  int * const vote_st_pnt = reinterpret_cast<int *>(rho_theta.data);

  for (int v = 0; v < rho_theta.rows; v++) {
    for (int u = 0; u < rho_theta.cols; u++) {

      double rho = v * rho_res;
      double theta = u * theta_res;
      int score = *(vote_st_pnt + v * rho_theta.cols + u);

      if (score <= score_thresh) {
        continue;
      }
      LineElem line2(rho, theta, score);
      LineElem line = line2;
      lines.push_back(line);
    }
  }

  std::sort(lines.begin(), lines.end());  

}

void HoughTrans::get_voted_img(cv::Mat &gray_img) {
  double min, max;
  cv::minMaxLoc(rho_theta, &min, &max);
  rho_theta.convertTo(gray_img, CV_8U, 256.0 / (max - min), - (256.0 * min) / (max - min));
}

void HoughTrans::vote_rho_theta(cv::Mat &gray_img) {

  // Initialize Buffer.
  rho_theta = 0;

  unsigned char * const img_pnt = gray_img.data;
  int * const vote_st_pnt = reinterpret_cast<int *>(rho_theta.data);

  // Outer loop for image 
  for (int v = 0; v < gray_img.rows; v++) {
    for (int u = 0; u < gray_img.cols; u++) {

      unsigned char val = *(img_pnt + v * gray_img.cols + u);

      if (val >= vote_thresh) {

        // Inner loop for voting
        for (int theta_col = 0;  theta_col < rho_theta.cols; theta_col++) {

          // Rho-Theta Calculation
          double theta = theta_res * theta_col * M_PI / 180.0;
          double rho_abs = std::abs(u * cos(theta) + v * sin(theta));


          if (rho_abs <= rho_max) {
            // Hough Vote
            int rho_row = std::abs(static_cast<int>(rho_abs / rho_res));
            int *tgt_pnt = vote_st_pnt + rho_row * rho_theta.cols + theta_col;
            *tgt_pnt = *tgt_pnt + 1;
          }

        }
      }
    }
  }
}

void HoughTrans::get_lines(cv::Mat &gray_img, std::vector<LineElem> &lines) {

  // 1. Vote
  vote_rho_theta(gray_img);

  // 2. Get Highly Ranked Lines
  get_top_ranked_lines(lines);

}
