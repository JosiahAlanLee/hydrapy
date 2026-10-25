// Recording fixture for the demonstration only; never shipped in GorgonBridge.
#include "plugin.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

struct DemoCapture : Module {
  FILE* pcm = nullptr;
  FILE* meters = nullptr;
  long samples = 0;
  int command = 0, pulse = 0;
  double stamp = 0.;
  std::string prefix;
  DemoCapture() {
    config(0, 9, 2);
    const char* path = std::getenv("GORGON_CAPTURE");
    if (path) prefix = path;
    configInput(0, "Polyphonic voice recording");
    configOutput(0, "Demonstration play trigger");
    configOutput(1, "Demonstration reset trigger");
  }
  ~DemoCapture() { if (pcm) fclose(pcm); if (meters) fclose(meters); }
  void process(const ProcessArgs& args) override {
    if (prefix.empty()) return;
    if (!pcm) {
      pcm = fopen((prefix + ".f32").c_str(), "wb");
      meters = fopen((prefix + ".csv").c_str(), "w");
      stamp = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      std::ofstream meta(prefix + ".json");
      meta << "{\"start\":" << std::fixed << stamp << ",\"sample_rate\":" << args.sampleRate << "}";
      if (meters) fprintf(meters, "time,clock,reset,run,gate,pitch,velocity,z,return_cv\n");
    }
    float sample = inputs[0].getVoltageSum() / 20.f;
    if (pcm) fwrite(&sample, sizeof(float), 1, pcm);
    if (samples % 800 == 0) {
      if (meters) {
        fprintf(meters, "%.6f", samples / double(args.sampleRate));
        for (int i=1;i<9;i++) fprintf(meters, ",%.6f", inputs[i].getVoltage());
        fprintf(meters, "\n");
      }
      std::ifstream flag(prefix + ".command"); int next=0; flag >> next;
      if (next != command) { command = next; pulse = int(args.sampleRate * .01); }
    }
    outputs[0].setVoltage(pulse > 0 && command % 10 == 1 ? 10.f : 0.f);
    outputs[1].setVoltage(pulse > 0 && command % 10 == 2 ? 10.f : 0.f);
    if (pulse > 0) pulse--;
    samples++;
  }
};
struct DemoCaptureWidget : ModuleWidget {
  DemoCaptureWidget(DemoCapture* m) {
    setModule(m); box.size = Vec(120,380);
    for(int i=0;i<9;i++) addInput(createInputCentered<PJ301MPort>(Vec(25+(i%3)*35,40+(i/3)*50),m,i));
    addOutput(createOutputCentered<PJ301MPort>(Vec(35,245),m,0));
    addOutput(createOutputCentered<PJ301MPort>(Vec(85,245),m,1));
  }
};
Model* modelDemoCapture = createModel<DemoCapture, DemoCaptureWidget>("DemoCapture");
