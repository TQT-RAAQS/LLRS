#ifndef COLLECTOR_H_
#define COLLECTOR_H_

#include "llrs-lib/PreProc.h"
#include "llrs-lib/Settings.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <dirent.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define NUM_TIMERS 15

#ifdef LOGGING_RUNTIME
#define START_TIMER(module) Util::Collector::get_instance()->start_timer(module)
#define END_TIMER(module) Util::Collector::get_instance()->end_timer(module)
#define GET_EXTERNAL_TIME(module, time)                                        \
    Util::Collector::get_instance()->get_external_time(module, time)
#else
#define START_TIMER(module)
#define END_TIMER(module)
#define GET_EXTERNAL_TIME(module, time)
#endif

namespace Util {
using TIMING_PAIR = std::pair<std::chrono::high_resolution_clock::time_point,
                              std::chrono::high_resolution_clock::time_point>;

class Collector {
  private:
    Collector(){};
    static Collector *p_instance;
    std::unordered_map<std::string, TIMING_PAIR> timers;
    long long elapsed_time(const std::string &module);

  public:
    Collector(Collector &other) = delete;
    void operator=(const Collector &) = delete;
    static Collector *get_instance();
    void start_timer(const std::string &module);
    void end_timer(const std::string &module);
    void get_external_time(const std::string &module, float time);
    std::vector<std::tuple<std::string, long long>> get_runtime_data();
    void clear_timers() {
        timers.clear();
        timers.reserve(NUM_TIMERS);
    }
};

} // namespace Util

template <typename T> static std::string vec_to_str(const std::vector<T> &vec) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto &elem : vec) {
        if (!first) {
            oss << ",";
        }
        oss << std::to_string(elem);
        first = false;
    }
    oss << "]";
    return oss.str();
}

template <typename Duration>
static long long microseconds_cast(Duration duration) {
    return std::chrono::duration_cast<std::chrono::microseconds>(duration)
        .count();
}

static std::chrono::high_resolution_clock::time_point HRCLOCK_NOW() {
    return std::chrono::high_resolution_clock::now();
}
#endif // COLLECTOR_H