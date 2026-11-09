#include "libobsensor/ObSensor.hpp"
#include "libobsensor/hpp/Error.hpp"

#include "../native/orbbec_profiles.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

struct Stats {
    std::atomic<uint64_t> frame_sets{0};
    std::atomic<uint64_t> color_frames{0};
    std::atomic<uint64_t> depth_frames{0};
    std::atomic<uint64_t> paired_frames{0};
    std::atomic<uint64_t> color_only_sets{0};
    std::atomic<uint64_t> depth_only_sets{0};
    std::atomic<uint64_t> first_color_ts{0};
    std::atomic<uint64_t> last_color_ts{0};
    std::atomic<uint64_t> first_depth_ts{0};
    std::atomic<uint64_t> last_depth_ts{0};
    std::atomic<uint32_t> color_width{0};
    std::atomic<uint32_t> color_height{0};
    std::atomic<uint32_t> depth_width{0};
    std::atomic<uint32_t> depth_height{0};
    std::mutex timing_mutex;
    std::chrono::steady_clock::time_point first_color_wall;
    std::chrono::steady_clock::time_point last_color_wall;
    std::chrono::steady_clock::time_point first_depth_wall;
    std::chrono::steady_clock::time_point last_depth_wall;
    double color_interval_sum_ms = 0.0;
    double depth_interval_sum_ms = 0.0;
    double max_color_interval_ms = 0.0;
    double max_depth_interval_ms = 0.0;
    uint64_t color_interval_count = 0;
    uint64_t depth_interval_count = 0;
};

struct Options {
    int duration_seconds = 30;
    vision_native::ColorFormat color_format = vision_native::ColorFormat::Mjpg;
};

Options parseOptions(int argc, char **argv) {
    Options options;
    for(int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if(index == 1 && argument.rfind("--", 0) != 0) {
            options.duration_seconds = std::max(1, std::atoi(argument.c_str()));
            continue;
        }
        if(argument == "--color-format") {
            if(++index >= argc) {
                throw std::invalid_argument("--color-format requires a value");
            }
            options.color_format = vision_native::parseColorFormat(argv[index]);
            continue;
        }
        if(argument == "--help" || argument == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [SECONDS] [--color-format mjpg|rgb|bgra]\n";
            std::exit(EXIT_SUCCESS);
        }
        throw std::invalid_argument("unknown option: " + argument);
    }
    return options;
}

void recordWallInterval(Stats &stats, bool color) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(stats.timing_mutex);
    auto &first = color ? stats.first_color_wall : stats.first_depth_wall;
    auto &last = color ? stats.last_color_wall : stats.last_depth_wall;
    auto &sum = color ? stats.color_interval_sum_ms : stats.depth_interval_sum_ms;
    auto &maximum = color ? stats.max_color_interval_ms : stats.max_depth_interval_ms;
    auto &count = color ? stats.color_interval_count : stats.depth_interval_count;
    if(count > 0) {
        const double interval = std::chrono::duration<double, std::milli>(now - last).count();
        sum += interval;
        maximum = std::max(maximum, interval);
        ++count;
    }
    else {
        first = now;
        ++count;
    }
    last = now;
}

void recordTimestamp(std::atomic<uint64_t> &first,
                     std::atomic<uint64_t> &last,
                     uint64_t timestamp) {
    uint64_t expected = 0;
    first.compare_exchange_strong(expected, timestamp);
    last.store(timestamp);
}

double rate(uint64_t count, uint64_t first, uint64_t last) {
    if(count < 2 || last <= first) {
        return 0.0;
    }
    return static_cast<double>(count - 1) * 1000000.0 /
           static_cast<double>(last - first);
}

}  // namespace

int main(int argc, char **argv) try {
    const auto options = parseOptions(argc, argv);
    Stats stats;

    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();
    config->enableStream(vision_native::chooseColorProfile(pipeline, options.color_format));
    config->enableVideoStream(OB_STREAM_DEPTH);

    pipeline.start(config, [&](std::shared_ptr<ob::FrameSet> frameset) {
        if(!frameset) {
            return;
        }

        auto color = frameset->colorFrame();
        auto depth = frameset->depthFrame();
        stats.frame_sets.fetch_add(1);

        if(color) {
            stats.color_frames.fetch_add(1);
            recordWallInterval(stats, true);
            stats.color_width.store(color->width());
            stats.color_height.store(color->height());
            recordTimestamp(stats.first_color_ts, stats.last_color_ts, color->timeStampUs());
        }
        if(depth) {
            stats.depth_frames.fetch_add(1);
            recordWallInterval(stats, false);
            stats.depth_width.store(depth->width());
            stats.depth_height.store(depth->height());
            recordTimestamp(stats.first_depth_ts, stats.last_depth_ts, depth->timeStampUs());
        }
        if(color && depth) {
            stats.paired_frames.fetch_add(1);
        }
        else if(color) {
            stats.color_only_sets.fetch_add(1);
        }
        else if(depth) {
            stats.depth_only_sets.fetch_add(1);
        }
    });

    const auto wallStart = std::chrono::steady_clock::now();
    std::cout << "Started color + depth pipeline for " << options.duration_seconds
              << " seconds, color format "
              << vision_native::colorFormatName(options.color_format) << ".\n";
    std::this_thread::sleep_for(std::chrono::seconds(options.duration_seconds));
    const auto wallEnd = std::chrono::steady_clock::now();
    pipeline.stop();

    const auto frameSets = stats.frame_sets.load();
    const auto colorFrames = stats.color_frames.load();
    const auto depthFrames = stats.depth_frames.load();
    const auto pairedFrames = stats.paired_frames.load();
    const auto firstColor = stats.first_color_ts.load();
    const auto lastColor = stats.last_color_ts.load();
    const auto firstDepth = stats.first_depth_ts.load();
    const auto lastDepth = stats.last_depth_ts.load();
    const double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
    double colorAverageInterval = 0.0;
    double depthAverageInterval = 0.0;
    double maxColorInterval = 0.0;
    double maxDepthInterval = 0.0;
    {
        std::lock_guard<std::mutex> lock(stats.timing_mutex);
        if(stats.color_interval_count > 1) {
            colorAverageInterval = stats.color_interval_sum_ms /
                                   (stats.color_interval_count - 1);
        }
        if(stats.depth_interval_count > 1) {
            depthAverageInterval = stats.depth_interval_sum_ms /
                                   (stats.depth_interval_count - 1);
        }
        maxColorInterval = stats.max_color_interval_ms;
        maxDepthInterval = stats.max_depth_interval_ms;
    }

    std::cout << "\n=== Astra multi-stream probe ===\n"
              << "frame sets:       " << frameSets << "\n"
              << "color frames:     " << colorFrames << " ("
              << stats.color_width.load() << "x" << stats.color_height.load() << ")\n"
              << "depth frames:     " << depthFrames << " ("
              << stats.depth_width.load() << "x" << stats.depth_height.load() << ")\n"
              << "paired frame sets: " << pairedFrames << "\n"
              << "color-only sets:   " << stats.color_only_sets.load() << "\n"
              << "depth-only sets:   " << stats.depth_only_sets.load() << "\n"
              << std::fixed << std::setprecision(2)
              << "wall duration:     " << wallSeconds << " s\n"
              << "color format:      "
              << vision_native::colorFormatName(options.color_format) << "\n"
              << "color wall FPS:    " << colorFrames / wallSeconds << "\n"
              << "depth wall FPS:    " << depthFrames / wallSeconds << "\n"
              << "color avg interval: " << colorAverageInterval << " ms\n"
              << "color max interval: " << maxColorInterval << " ms\n"
              << "depth avg interval: " << depthAverageInterval << " ms\n"
              << "depth max interval: " << maxDepthInterval << " ms\n"
              << "color timestamp FPS: unavailable (Astra color clock)\n"
              << "depth timestamp FPS: " << rate(depthFrames, firstDepth, lastDepth) << "\n";

    const bool success = colorFrames >= 10 && depthFrames >= 10 && pairedFrames >= 10;
    std::cout << "result: " << (success ? "PASS" : "FAIL") << "\n";
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch(const ob::Error &e) {
    std::cerr << "Orbbec SDK error\n"
              << "function: " << e.getName() << "\n"
              << "args: " << e.getArgs() << "\n"
              << "message: " << e.getMessage() << "\n"
              << "type: " << e.getExceptionType() << "\n";
    return EXIT_FAILURE;
}
catch(const std::exception &e) {
    std::cerr << "Probe error: " << e.what() << "\n";
    return EXIT_FAILURE;
}
