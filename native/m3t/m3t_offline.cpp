// Project-local offline adapter. M3T remains in third_party, unmodified.
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/tracker.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
class RecordedCamera : public m3t::ColorCamera {
 public:
  RecordedCamera(const m3t::Intrinsics &intrinsics) : ColorCamera("recorded_rgb") {
    intrinsics_=intrinsics;
  }
  bool SetUp() override { set_up_=!image_.empty(); return set_up_; }
  bool UpdateImage(bool) override { return !image_.empty(); }
  void SetFrame(const cv::Mat &frame) { image_=frame; }
};
int main(int argc,char **argv) {
 try {
  if(argc!=2) throw std::runtime_error("Usage: m3t_offline config.yml");
  cv::setNumThreads(1);
  cv::FileStorage cfg(argv[1],cv::FileStorage::READ);
  if(!cfg.isOpened()) throw std::runtime_error("Cannot open config");
  auto text=[&](const char *key){return (std::string)cfg[key];};
  cv::Mat K,dist,T; cfg["camera_matrix"]>>K; cfg["distortion"]>>dist; cfg["initial_pose_m"]>>T;
  if(K.rows!=3 || K.cols!=3 || T.rows!=4 || T.cols!=4) throw std::runtime_error("Invalid calibration / initial pose");
  cv::VideoCapture cap(text("video"));
  int start_frame=(int)cfg["start_frame"];
  if(start_frame) cap.set(cv::CAP_PROP_POS_FRAMES,start_frame);
  cv::Mat raw,frame,map1,map2;
  if(!cap.read(raw)) throw std::runtime_error("Cannot read first video frame");
  cv::initUndistortRectifyMap(K,dist,cv::Mat(),K,raw.size(),CV_32FC1,map1,map2);
  cv::remap(raw,frame,map1,map2,cv::INTER_LINEAR);
  m3t::Intrinsics intr{(float)K.at<double>(0,0),(float)K.at<double>(1,1),
      (float)K.at<double>(0,2),(float)K.at<double>(1,2),raw.cols,raw.rows};
  auto camera=std::make_shared<RecordedCamera>(intr); camera->SetFrame(frame);
  auto body=std::make_shared<m3t::Body>("energy_unit",text("mesh"),.001f,true,false,m3t::Transform3fA::Identity());
  m3t::Transform3fA pose=m3t::Transform3fA::Identity();
  for(int r=0;r<4;r++) for(int c=0;c<4;c++) pose.matrix()(r,c)=T.at<double>(r,c);
  body->set_body2world_pose(pose);
  auto model=std::make_shared<m3t::RegionModel>("energy_region",body,text("model_cache"),
      .8f,(int)cfg["n_divides"],200,.05f,.002f,false,(int)cfg["model_image_size"]);
  auto region=std::make_shared<m3t::RegionModality>("energy_region",body,camera,model);
  auto link=std::make_shared<m3t::Link>("energy_link",body); link->AddModality(region);
  auto optimizer=std::make_shared<m3t::Optimizer>("energy_optimizer",link);
  auto tracker=std::make_shared<m3t::Tracker>("offline"); tracker->AddOptimizer(optimizer);
  auto setup_start=Clock::now();
  if(!tracker->SetUp()) throw std::runtime_error("M3T setup failed (check OpenGL/display/dependencies)");
  double setup_ms=milliseconds(setup_start);
  tracker->StartTracking();
  if(!tracker->ExecuteStartingStep(0)) throw std::runtime_error("M3T start failed");
  std::ofstream csv(text("poses_csv"));
  if(!csv) throw std::runtime_error("Cannot create CSV");
  csv<<"frame_index,api_ok,processing_ms,rx,ry,rz,tx_mm,ty_mm,tz_mm\n"<<std::setprecision(10);
  int index=0,max_frames=(int)cfg["max_frames"]; double total_ms=0;
  do {
    auto start=Clock::now();
    if(index) {
      cv::remap(raw,frame,map1,map2,cv::INTER_LINEAR); camera->SetFrame(frame);
    }
    bool ok=index==0 || tracker->ExecuteTrackingStep(index);
    double elapsed=milliseconds(start); total_ms+=elapsed;
    auto transform=body->body2world_pose();
    cv::Mat rotation(3,3,CV_64F),rvec;
    for(int r=0;r<3;r++) for(int c=0;c<3;c++) rotation.at<double>(r,c)=transform.matrix()(r,c);
    cv::Rodrigues(rotation,rvec);
    csv<<(start_frame+index)<<","<<ok<<","<<elapsed;
    for(int j=0;j<3;j++) csv<<","<<rvec.at<double>(j);
    for(int j=0;j<3;j++) csv<<","<<transform.translation()(j)*1000.;
    csv<<"\n";
    if(!ok) throw std::runtime_error("M3T tracking API failed at frame "+std::to_string(index));
    ++index;
    if(index%250==0) std::cerr<<"Processed "<<index<<" frames\n";
    if(max_frames>0 && index>=max_frames) break;
  } while(cap.read(raw));
  cv::FileStorage result(text("native_summary"),cv::FileStorage::WRITE);
  result<<"frames"<<index<<"setup_ms"<<setup_ms<<"mean_processing_ms"<<(total_ms/std::max(1,index));
  std::cout<<"Completed "<<index<<" frames; mean "<<total_ms/std::max(1,index)<<" ms; setup "<<setup_ms<<" ms\n";
  return 0;
 } catch(const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
}
