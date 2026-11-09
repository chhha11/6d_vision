#include "libobsensor/ObSensor.hpp"
#include "libobsensor/hpp/Error.hpp"

#include "orbbec_profiles.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path output = "data/rgbd_capture/current";
    int seconds = 20;
    uint64_t pairs = 30;
    vision_native::ColorFormat color_format = vision_native::ColorFormat::Mjpg;
};

struct Counters {
    std::atomic<uint64_t> frameSets{0};
    std::atomic<uint64_t> colorFrames{0};
    std::atomic<uint64_t> depthFrames{0};
    std::atomic<uint64_t> pairedFrames{0};
    std::atomic<uint64_t> emptyDepthFrames{0};
    std::atomic<uint64_t> savedPairs{0};
    std::atomic<uint64_t> callbackErrors{0};
};

std::string usage(const char *program) {
    std::ostringstream out;
    out << "Usage: " << program << " [options]\n"
        << "  --output DIR    Capture directory (default: data/rgbd_capture/current)\n"
        << "  --seconds N     Maximum capture duration (default: 20)\n"
        << "  --pairs N       Number of paired samples to save (default: 30)\n"
        << "  --color-format F  Color format: mjpg, rgb, or bgra (default: mjpg)\n"
        << "  --help          Show this help\n";
    return out.str();
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const char *name) -> std::string {
            if(i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if(arg == "--output") {
            options.output = requireValue("--output");
        }
        else if(arg == "--seconds") {
            options.seconds = std::stoi(requireValue("--seconds"));
        }
        else if(arg == "--pairs") {
            options.pairs = std::stoull(requireValue("--pairs"));
        }
        else if(arg == "--color-format") {
            options.color_format =
                vision_native::parseColorFormat(requireValue("--color-format"));
        }
        else if(arg == "--help" || arg == "-h") {
            std::cout << usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if(options.seconds <= 0 || options.pairs == 0) {
        throw std::invalid_argument("--seconds and --pairs must be positive");
    }
    return options;
}

std::string frameStem(uint64_t index) {
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << index;
    return name.str();
}

cv::Mat decodeColor(const std::shared_ptr<ob::ColorFrame> &frame) {
    const int width = static_cast<int>(frame->width());
    const int height = static_cast<int>(frame->height());
    const auto format = frame->format();

    if(format == OB_FORMAT_MJPG) {
        const auto *bytes = static_cast<const unsigned char *>(frame->data());
        cv::Mat encoded(1, static_cast<int>(frame->dataSize()), CV_8UC1,
                        const_cast<unsigned char *>(bytes));
        return cv::imdecode(encoded, cv::IMREAD_COLOR);
    }
    if(format == OB_FORMAT_BGR) {
        return cv::Mat(height, width, CV_8UC3, frame->data()).clone();
    }
    if(format == OB_FORMAT_RGB) {
        cv::Mat rgb(height, width, CV_8UC3, frame->data());
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }
    if(format == OB_FORMAT_BGRA) {
        cv::Mat bgra(height, width, CV_8UC4, frame->data());
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }

    std::ostringstream message;
    message << "Unsupported color format: " << static_cast<int>(format);
    throw std::runtime_error(message.str());
}

cv::Mat copyDepth(const std::shared_ptr<ob::DepthFrame> &frame) {
    const int width = static_cast<int>(frame->width());
    const int height = static_cast<int>(frame->height());
    const size_t expected = static_cast<size_t>(width) * height * sizeof(uint16_t);
    if(frame->data() == nullptr || frame->dataSize() < expected) {
        std::ostringstream message;
        message << "Depth frame has insufficient data: " << frame->dataSize()
                << " bytes, expected at least " << expected;
        throw std::runtime_error(message.str());
    }
    return cv::Mat(height, width, CV_16UC1, frame->data()).clone();
}

void writeMetadataHeader(std::ofstream &metadata) {
    metadata << "pair_index,color_file,depth_file,color_timestamp_us,depth_timestamp_us,"
                "color_system_timestamp_us,depth_system_timestamp_us,frame_timestamp_delta_us,"
                "system_timestamp_delta_us,color_width,color_height,depth_width,"
                "depth_height,depth_value_scale\n";
}

void writeCameraParams(const fs::path &output, const OBCameraParam &params) {
    std::ofstream metadata(output / "camera_params.json", std::ios::out | std::ios::trunc);
    if(!metadata) {
        throw std::runtime_error("Cannot open camera parameter metadata file");
    }
    const auto writeIntrinsic = [&](const char *name, const OBCameraIntrinsic &intrinsic) {
        metadata << "  \"" << name << "\": {\n"
                 << "    \"fx\": " << intrinsic.fx << ",\n"
                 << "    \"fy\": " << intrinsic.fy << ",\n"
                 << "    \"cx\": " << intrinsic.cx << ",\n"
                 << "    \"cy\": " << intrinsic.cy << ",\n"
                 << "    \"width\": " << intrinsic.width << ",\n"
                 << "    \"height\": " << intrinsic.height << "\n"
                 << "  }";
    };
    metadata << "{\n";
    writeIntrinsic("depth_intrinsic", params.depthIntrinsic);
    metadata << ",\n";
    writeIntrinsic("color_intrinsic", params.rgbIntrinsic);
    metadata << "\n}\n";
}

void removeIndexedFrames(const fs::path &directory, const char *extension) {
    const std::regex indexedName(R"(^[0-9]{6}\.)" + std::string(extension) + R"($)");
    std::error_code error;
    for(const auto &entry : fs::directory_iterator(directory, error)) {
        if(error) {
            throw std::system_error(error, "Cannot enumerate capture directory");
        }
        if(!entry.is_regular_file(error)) {
            if(error) {
                throw std::system_error(error, "Cannot inspect capture file");
            }
            continue;
        }
        if(!std::regex_match(entry.path().filename().string(), indexedName)) {
            continue;
        }
        fs::remove(entry.path(), error);
        if(error) {
            throw std::system_error(error, "Cannot remove stale capture frame");
        }
    }
}

}  // namespace

int main(int argc, char **argv) try {
    const Options options = parseOptions(argc, argv);
    const fs::path colorDir = options.output / "color";
    const fs::path depthDir = options.output / "depth";
    fs::create_directories(colorDir);
    fs::create_directories(depthDir);
    // A reused output directory must not leave frames beyond the new CSV row
    // count, otherwise the capture appears longer than its declared pairs.
    removeIndexedFrames(colorDir, "jpg");
    removeIndexedFrames(depthDir, "png");

    std::ofstream metadata(options.output / "pairs.csv", std::ios::out | std::ios::trunc);
    if(!metadata) {
        throw std::runtime_error("Cannot open capture metadata file");
    }
    writeMetadataHeader(metadata);

    Counters counters;
    std::atomic<bool> stopRequested{false};
    std::mutex writeMutex;
    std::mutex errorMutex;
    std::string callbackError;

    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();
    config->enableStream(
        vision_native::chooseColorProfile(pipeline, options.color_format));
    config->enableVideoStream(OB_STREAM_DEPTH);

    pipeline.start(config, [&](std::shared_ptr<ob::FrameSet> frameset) {
        if(!frameset) {
            return;
        }

        counters.frameSets.fetch_add(1);
        auto color = frameset->colorFrame();
        auto depth = frameset->depthFrame();
        if(color) {
            counters.colorFrames.fetch_add(1);
        }
        if(depth) {
            counters.depthFrames.fetch_add(1);
        }
        if(!color || !depth) {
            return;
        }
        counters.pairedFrames.fetch_add(1);

        std::lock_guard<std::mutex> lock(writeMutex);
        if(stopRequested.load() || counters.savedPairs.load() >= options.pairs) {
            return;
        }

        const uint64_t pairIndex = counters.savedPairs.fetch_add(1);
        const std::string stem = frameStem(pairIndex);
        const fs::path colorPath = colorDir / (stem + ".jpg");
        const fs::path depthPath = depthDir / (stem + ".png");

        try {
            cv::Mat colorImage = decodeColor(color);
            cv::Mat depthImage = copyDepth(depth);
            if(colorImage.empty()) {
                throw std::runtime_error("Cannot decode color frame");
            }
            // Some Astra units emit an all-zero depth buffer while the depth
            // stream is warming up. It is still a valid SDK frame object, but
            // it is not useful to the RGB-D fitter and must not be saved as a
            // successful pair.
            if(cv::countNonZero(depthImage) == 0) {
                counters.emptyDepthFrames.fetch_add(1);
                counters.savedPairs.fetch_sub(1);
                return;
            }
            if(!cv::imwrite(colorPath.string(), colorImage,
                            {cv::IMWRITE_JPEG_QUALITY, 95})) {
                throw std::runtime_error("Cannot write color image");
            }
            if(!cv::imwrite(depthPath.string(), depthImage,
                            {cv::IMWRITE_PNG_COMPRESSION, 1})) {
                throw std::runtime_error("Cannot write depth image");
            }

            const auto colorTimestamp = color->timeStampUs();
            const auto depthTimestamp = depth->timeStampUs();
            const auto colorSystemTimestamp = color->systemTimeStampUs();
            const auto depthSystemTimestamp = depth->systemTimeStampUs();
            // The two sensor timestamp domains are different on this Astra. The
            // SDK system timestamps are comparable, while FrameSet membership is
            // the actual pairing decision.
            const int64_t timestampDelta =
                static_cast<int64_t>(colorTimestamp) - static_cast<int64_t>(depthTimestamp);
            const int64_t systemTimestampDelta =
                static_cast<int64_t>(colorSystemTimestamp) -
                static_cast<int64_t>(depthSystemTimestamp);

            metadata << pairIndex << ',' << colorPath.filename().string() << ','
                     << depthPath.filename().string() << ',' << colorTimestamp << ','
                     << depthTimestamp << ',' << colorSystemTimestamp << ','
                     << depthSystemTimestamp << ',' << timestampDelta << ','
                     << systemTimestampDelta << ','
                     << color->width() << ',' << color->height() << ','
                     << depth->width() << ',' << depth->height() << ','
                     << depth->getValueScale() << '\n';
            metadata.flush();

            if(counters.savedPairs.load() >= options.pairs) {
                stopRequested.store(true);
            }
        }
        catch(const std::exception &error) {
            std::error_code ignored;
            fs::remove(colorPath, ignored);
            fs::remove(depthPath, ignored);
            counters.savedPairs.fetch_sub(1);
            counters.callbackErrors.fetch_add(1);
            {
                std::lock_guard<std::mutex> errorLock(errorMutex);
                if(callbackError.empty()) {
                    callbackError = error.what();
                }
            }
            stopRequested.store(true);
        }
        catch(...) {
            std::error_code ignored;
            fs::remove(colorPath, ignored);
            fs::remove(depthPath, ignored);
            counters.savedPairs.fetch_sub(1);
            counters.callbackErrors.fetch_add(1);
            {
                std::lock_guard<std::mutex> errorLock(errorMutex);
                if(callbackError.empty()) {
                    callbackError = "unknown callback error";
                }
            }
            stopRequested.store(true);
        }
    });

    // Persist the actual profile intrinsics used by this native pipeline. The
    // Python depth initializer rejects captures whose geometry is not tied to
    // the same calibrated 640x480 depth profile.
    writeCameraParams(options.output, pipeline.getCameraParam());

    const auto started = std::chrono::steady_clock::now();
    while(!stopRequested.load()) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - started);
        if(elapsed.count() >= options.seconds) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    pipeline.stop();

    const auto frameSets = counters.frameSets.load();
    const auto colorFrames = counters.colorFrames.load();
    const auto depthFrames = counters.depthFrames.load();
    const auto pairedFrames = counters.pairedFrames.load();
    const auto emptyDepthFrames = counters.emptyDepthFrames.load();
    const auto savedPairs = counters.savedPairs.load();
    const auto callbackErrors = counters.callbackErrors.load();

    std::ofstream summary(options.output / "summary.txt", std::ios::out | std::ios::trunc);
    summary << "frame_sets=" << frameSets << '\n'
            << "color_frames=" << colorFrames << '\n'
            << "depth_frames=" << depthFrames << '\n'
            << "paired_frames=" << pairedFrames << '\n'
            << "empty_depth_frames=" << emptyDepthFrames << '\n'
            << "saved_pairs=" << savedPairs << '\n'
            << "callback_errors=" << callbackErrors << '\n'
            << "requested_pairs=" << options.pairs << '\n'
            << "requested_seconds=" << options.seconds << '\n';
    if(!callbackError.empty()) {
        summary << "first_callback_error=" << callbackError << '\n';
    }

    std::cout << "Saved RGB-D capture to " << options.output << "\n"
              << "frame sets: " << frameSets << "\n"
              << "color frames: " << colorFrames << "\n"
              << "depth frames: " << depthFrames << "\n"
              << "paired frames: " << pairedFrames << "\n"
              << "empty depth frames: " << emptyDepthFrames << "\n"
              << "saved pairs: " << savedPairs << "\n"
              << "callback errors: " << callbackErrors << "\n";
    if(!callbackError.empty()) {
        std::cout << "first callback error: " << callbackError << "\n";
    }

    return savedPairs > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch(const ob::Error &e) {
    std::cerr << "Orbbec SDK error\n"
              << "function: " << e.getName() << '\n'
              << "args: " << e.getArgs() << '\n'
              << "message: " << e.getMessage() << '\n'
              << "type: " << e.getExceptionType() << '\n';
    return EXIT_FAILURE;
}
catch(const std::exception &e) {
    std::cerr << "Capture error: " << e.what() << '\n';
    return EXIT_FAILURE;
}
