// Persistent camera-free M3T adapter. All stdout diagnostics go to stderr;
// the duplicated stdout descriptor is reserved for the binary protocol.
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/tracker.h>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

class StreamCamera : public m3t::ColorCamera {
 public:
  StreamCamera(const m3t::Intrinsics &intr) : ColorCamera("live_rgb") {
    intrinsics_=intr; image_=cv::Mat::zeros(intr.height, intr.width, CV_8UC3);
  }
  bool SetUp() override { set_up_=true; return true; }
  bool UpdateImage(bool) override { return !image_.empty(); }
  void SetFrame(const cv::Mat &frame) { image_=frame; }
};
int main(int argc, char **argv) {
  FILE *output=fdopen(dup(STDOUT_FILENO), "wb");
  dup2(STDERR_FILENO, STDOUT_FILENO);
  try {
    if(argc!=2 || !output) throw std::runtime_error("Usage: m3t_stream config.yml");
    cv::setNumThreads(1);
    cv::FileStorage cfg(argv[1], cv::FileStorage::READ);
    if(!cfg.isOpened()) throw std::runtime_error("Cannot open stream config");
    cv::Mat K,dist; cfg["camera_matrix"]>>K; cfg["distortion"]>>dist;
    int width=(int)cfg["width"], height=(int)cfg["height"];
    if(K.type()!=CV_64F || K.rows!=3 || K.cols!=3 || width<1 || height<1)
      throw std::runtime_error("Invalid stream calibration");
    auto text=[&](const char *key){return (std::string)cfg[key];};
    cv::Mat map1,map2;
    cv::initUndistortRectifyMap(K,dist,cv::Mat(),K,{width,height},CV_32FC1,map1,map2);
    m3t::Intrinsics intr{(float)K.at<double>(0,0),(float)K.at<double>(1,1),
      (float)K.at<double>(0,2),(float)K.at<double>(1,2),width,height};
    auto camera=std::make_shared<StreamCamera>(intr);
    auto body=std::make_shared<m3t::Body>("energy_unit",text("mesh"),.001f,true,false,m3t::Transform3fA::Identity());
    auto pose=m3t::Transform3fA::Identity(); pose.translation().z()=.5f; body->set_body2world_pose(pose);
    auto model=std::make_shared<m3t::RegionModel>("energy_region",body,text("model_cache"),
      .8f,3,200,.05f,.002f,false,1000);
    auto region=std::make_shared<m3t::RegionModality>("energy_region",body,camera,model);
    auto link=std::make_shared<m3t::Link>("energy_link",body); link->AddModality(region);
    auto optimizer=std::make_shared<m3t::Optimizer>("energy_optimizer",link);
    auto tracker=std::make_shared<m3t::Tracker>("live"); tracker->AddOptimizer(optimizer);
    if(!tracker->SetUp()) throw std::runtime_error("M3T setup failed");
    tracker->StartTracking();
    const uint32_t magic=0x4d335431;
    if(fwrite(&magic,sizeof(magic),1,output)!=1 || fflush(output)) return 1;
    cv::Mat raw(height,width,CV_8UC3),frame;
    bool initialized=false;
    while(true) {
      uint32_t header[2];
      if(fread(header,sizeof(header),1,stdin)!=1 || !header[0]) break;
      if(header[0]!=1 && header[0]!=2) throw std::runtime_error("Invalid stream command");
      double params[6];
      if(fread(params,sizeof(params),1,stdin)!=1 ||
         fread(raw.data,raw.total()*raw.elemSize(),1,stdin)!=1) break;
      auto started=std::chrono::steady_clock::now();
      cv::remap(raw,frame,map1,map2,cv::INTER_LINEAR); camera->SetFrame(frame);
      bool ok=false;
      if(header[0]==1) {
        cv::Mat rotation; cv::Rodrigues(cv::Mat(3,1,CV_64F,params),rotation);
        auto reset=m3t::Transform3fA::Identity();
        for(int r=0;r<3;r++) {
          reset.translation()(r)=(float)(params[3+r]/1000.);
          for(int c=0;c<3;c++) reset.linear()(r,c)=(float)rotation.at<double>(r,c);
        }
        body->set_body2world_pose(reset);
        initialized=tracker->ExecuteStartingStep(header[1]); ok=initialized;
      } else if(initialized) ok=tracker->ExecuteTrackingStep(header[1]);
      double result[7];
      const auto transform=body->body2world_pose(); cv::Mat rotation(3,3,CV_64F),rvec;
      for(int r=0;r<3;r++) for(int c=0;c<3;c++) rotation.at<double>(r,c)=transform.linear()(r,c);
      cv::Rodrigues(rotation,rvec);
      for(int r=0;r<3;r++) { result[r]=rvec.at<double>(r); result[3+r]=transform.translation()(r)*1000.; }
      result[6]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      uint32_t response[2]={header[1],(uint32_t)ok};
      if(fwrite(response,sizeof(response),1,output)!=1 || fwrite(result,sizeof(result),1,output)!=1 || fflush(output)) break;
    }
    fclose(output); return 0;
  } catch(const std::exception &exc) { std::cerr<<exc.what()<<"\n"; return 1; }
}
