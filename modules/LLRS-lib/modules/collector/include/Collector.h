#ifndef COLLECTOR_H
#define COLLECTOR_H

#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
#include <algorithm>
#include <ctime>
#include <jsoncpp/json/json.h>
#include <LLRS-lib/Settings.h>
#include <LLRS-lib/PreProc.h>

template <typename T>
static std::string vec_to_str(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& elem : vec) {
        if (!first) {
            oss << ",";
        }
        oss << std::to_string(elem);
        first = false;
    }
    oss << "]";
    return oss.str();
}

template <typename Duration> static long long microseconds_cast(Duration duration) {
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

static std::chrono::high_resolution_clock::time_point HRCLOCK_NOW() {
    return std::chrono::high_resolution_clock::now();
}

namespace Util {

    using TIMING_PAIR = std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>;
    using CYCLE_INFO = std::vector<TIMING_PAIR>;
    using REP_INFO = std::vector<CYCLE_INFO>;
    using TRIAL_INFO = std::vector<REP_INFO>;

    class Collector {
    private:
        Collector(){};
        static Collector * p_instance;
        std::unordered_map<std::string, TRIAL_INFO> _timers;
    public:
        Collector(Collector &other) = delete;
        void operator=(const Collector &) = delete;
        static Collector * get_instance();
        void start_timer(const std::string& module, int trial, int rep, int cycle);
        void end_timer(const std::string& module, int trial, int rep, int cycle);
        void get_external_time(const std::string& module, int trial, int rep, int cycle, float time);
        long long elapsed_time(const std::string& module, int trial, int rep, int cycle);

        Json::Value gen_runtime_json();
    };
    
}

#endif // COLLECTOR_H