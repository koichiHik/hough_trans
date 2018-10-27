
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <set>


#include "opencv2/opencv.hpp"

#include "hough_trans.h"

void prepare_hough(HoughTrans &hough) {

  HoughTransParams param;
  param.rho_max = 400;
  param.theta_max = 180.0;
  param.rho_res = 1.0;
  param.theta_res = 0.5;
  param.vote_thresh = 1;
  param.score_thresh = 1;

  hough.initialize(param);

}

void calc_pnt_in_img(double rho, double theta, cv::Point2i &p1, cv::Point2i &p2) {

  int x1, y1, x2, y2;
  double theta_rad = theta * M_PI / 180.0;
  if ((theta < 45.0) || (135.0 < theta)) {
    y1 = -10000;
    y2 = 10000;
    x1 = -(sin(theta_rad)/cos(theta_rad)) * y1 + rho / cos(theta_rad);
    x2 = -(sin(theta_rad)/cos(theta_rad)) * y2 + rho / cos(theta_rad);
  } else {
    x1 = -10000;
    x2 = 10000;
    y1 = -(cos(theta_rad)/sin(theta_rad)) * x1 + rho / sin(theta_rad);
    y2 = -(cos(theta_rad)/sin(theta_rad)) * x2 + rho / sin(theta_rad);
  }
  p1.x = x1;
  p1.y = y1;
  p2.x = x2;
  p2.y = y2;
}

void recv_file_path(std::string &path) {

  do {
    std::cin >> path;
    std::ifstream ifs(path); 

    if (ifs.is_open()) {
      break;
    }

    std::cout << "The specified file can not be opened. Please enter again. " << std::endl;

  } while (true);
  
}

void recv_console_input(std::string &path) {

  std::cout << "Please enter image path." << std::endl;
  recv_file_path(path);
  return;
}


int main(int argc, char** argv) {

  std::cout << "Hough Trans Started!" << std::endl;

  
  // 0. Create Hough Object.
  std::string path;
  recv_console_input(path);
  HoughTrans hough;

  // 1. Prepare Hough Object.
  prepare_hough(hough);

  // 2. Load Image
  cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
  cv::Mat cImg = cv::imread(path, cv::IMREAD_UNCHANGED);

  // 3. Show Image
  cv::imshow("Sample", img);
  cv::imwrite("init.png", img);
  cv::waitKey(0);

  // 4. Hough Voting.
  std::vector<LineElem> lines;
  cv::threshold(img, img, 50, 255, cv::THRESH_BINARY);
  cv::bitwise_not(img, img);
  cv::imshow("Binarized", img);
  cv::imwrite("binarized.png", img);
  cv::waitKey(0);
  hough.get_lines(img, lines);

  // 5. Draw Line in image.
  for (int i = 0; i < 5; i++) {
    cv::Point2i pt1, pt2;
    calc_pnt_in_img(lines[i].rho, lines[i].theta, pt1, pt2);

    std::cout << "rho : " << lines[i].rho << ", theta : " << lines[i].theta << std::endl;
    std::cout << "pt1 : " << pt1 << ", pt2 : " << pt2 << std::endl;

    cv::line(cImg, pt1, pt2, cv::Scalar(255, 0, 0));
  }

  cv::Mat res;
  hough.get_voted_img(res);
  cv::imshow("Hough Vote", res);
  cv::imwrite("vote_res.png", res);
  cv::imshow("Result", cImg);
  cv::imwrite("res.png", cImg);
  cv::waitKey(0);

  return 0;
}
