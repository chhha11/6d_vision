#pragma once

#include "libobsensor/ObSensor.hpp"
#include "libobsensor/hpp/Error.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace vision_native {

enum class ColorFormat {
    Mjpg,
    Rgb,
    Bgra,
};

inline const char *colorFormatName(ColorFormat format) {
    switch(format) {
    case ColorFormat::Mjpg:
        return "mjpg";
    case ColorFormat::Rgb:
        return "rgb";
    case ColorFormat::Bgra:
        return "bgra";
    }
    return "unknown";
}

inline OBFormat colorFormatValue(ColorFormat format) {
    switch(format) {
    case ColorFormat::Mjpg:
        return OB_FORMAT_MJPG;
    case ColorFormat::Rgb:
        return OB_FORMAT_RGB;
    case ColorFormat::Bgra:
        return OB_FORMAT_BGRA;
    }
    throw std::invalid_argument("unknown color format");
}

inline ColorFormat parseColorFormat(const std::string &value) {
    if(value == "mjpg" || value == "mjpeg") {
        return ColorFormat::Mjpg;
    }
    if(value == "rgb") {
        return ColorFormat::Rgb;
    }
    if(value == "bgra") {
        return ColorFormat::Bgra;
    }
    throw std::invalid_argument(
        "invalid color format '" + value + "' (expected mjpg, rgb, or bgra)");
}

inline std::shared_ptr<ob::StreamProfile> asStreamProfile(
    const std::shared_ptr<ob::StreamProfile> &profile) {
    return std::const_pointer_cast<ob::StreamProfile>(profile);
}

inline std::shared_ptr<ob::StreamProfile> chooseColorProfile(
    ob::Pipeline &pipeline, ColorFormat requested = ColorFormat::Mjpg) {
    auto profiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    if(!profiles || profiles->count() == 0) {
        throw std::runtime_error("Astra color profile list is empty");
    }

    std::cout << "Available color profiles:\n";
    for(uint32_t index = 0; index < profiles->count(); ++index) {
        auto profile = profiles->getProfile(index);
        try {
            auto video = profile->as<ob::VideoStreamProfile>();
            std::cout << "  " << video->width() << "x" << video->height()
                      << " @ " << video->fps() << " FPS, format "
                      << static_cast<int>(video->format()) << "\n";
        }
        catch(const ob::Error &) {
            // Ignore profiles that are not video profiles.
        }
        catch(const std::exception &) {
            // Ignore non-video profiles if a device exposes any.
        }
    }

    const auto requestedValue = colorFormatValue(requested);
    for(uint32_t index = 0; index < profiles->count(); ++index) {
        auto candidate = profiles->getProfile(index);
        try {
            auto video = candidate->as<ob::VideoStreamProfile>();
            if(video->width() == 640 && video->height() == 480 &&
               video->fps() == 30 && video->format() == requestedValue) {
                std::cout << "Selected color profile: " << video->width()
                          << "x" << video->height() << " @ " << video->fps()
                          << " FPS, format " << colorFormatName(requested)
                          << " (" << static_cast<int>(video->format()) << ")\n";
                return asStreamProfile(candidate);
            }
        }
        catch(const ob::Error &) {
            // Ignore profiles that cannot be viewed as video profiles.
        }
        catch(const std::exception &) {
            // Ignore non-video profiles if a device exposes any.
        }
    }

    throw std::runtime_error(
        "Astra does not advertise 640x480 @ 30 FPS color format " +
        std::string(colorFormatName(requested)));
}

}  // namespace vision_native
