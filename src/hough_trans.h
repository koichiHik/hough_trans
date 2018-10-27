
#include "opencv2/opencv.hpp"
#include <vector>

class HoughTransParams {
public:
  int rho_max;
  int theta_max;
  int score_thresh;
  double rho_res;
  double theta_res;
  double vote_thresh;
};

class LineElem {

public:

  LineElem() : 
    rho(0.0), theta(0.0), score(0) {}

  LineElem(const LineElem &obj) :
    rho(obj.rho), theta(obj.theta), score(obj.score){}

  LineElem(double rho, double theta, int score) :
    rho(rho), theta(theta), score(score){
  }

  LineElem &operator=(const LineElem &obj) {
    if (this != &obj) {
      this->rho = obj.rho;
      this->theta = obj.theta;
      this->score = obj.score;
    }
    return *this;
  }

  bool operator<(const LineElem &other) const {
    return score > other.score;
  }

  double rho;
  double theta;
  int score;
};

class HoughTrans {

public:
  HoughTrans();

  ~HoughTrans();

  void initialize(const HoughTransParams &params);

  void get_lines(cv::Mat &gray_img, std::vector<LineElem> &lines);

  void get_voted_img(cv::Mat &gray_img);

private:

  void get_top_ranked_lines(std::vector<LineElem> &lines);

  void vote_rho_theta(cv::Mat &gray_img);

private:

  int rho_max, theta_max;
  int rho_theta_rows, rho_theta_cols;
  int score_thresh;
  double rho_res, theta_res, vote_thresh;

  cv::Mat rho_theta;

};
