#include "Collector.h"

Util::Collector *Util::Collector::p_instance{nullptr};

Util::Collector *Util::Collector::get_instance() {
    if (p_instance == nullptr) {
        p_instance = new Collector();
        p_instance->timers.reserve(NUM_TIMERS);
    }
    return p_instance;
}

void Util::Collector::get_external_time(const std::string &module, float time) {

    std::chrono::nanoseconds external_time{static_cast<int>(round(time))};
    auto time_now = std::chrono::high_resolution_clock::now();
    auto time_now_plus_alg_elapse = time_now + external_time;

    timers[module] =
        std::make_pair(time_now, time_now_plus_alg_elapse);
}

void Util::Collector::start_timer(const std::string &module) {
    auto time_now = std::chrono::high_resolution_clock::now();
    timers[module] = std::make_pair(time_now, time_now);
}

void Util::Collector::end_timer(const std::string &module) {
    auto finish = std::chrono::high_resolution_clock::now();
    timers[module].second = finish;
}

long long Util::Collector::elapsed_time(const std::string &module) {
    if (timers.find(module) == timers.end()) {
        return 0;
    }
    auto cycle_pair = timers[module];
    return std::chrono::duration_cast<std::chrono::nanoseconds>(cycle_pair.second - cycle_pair.first)
        .count();
}

std::vector<std::tuple<std::string, long long>> Util::Collector::get_runtime_data() {
    std::vector<std::tuple<std::string, long long>> runtime_data;
    runtime_data.reserve(NUM_TIMERS);
    for (const auto &timer : timers) {
        runtime_data.push_back(std::make_tuple(timer.first, elapsed_time(timer.first)));
    }
    return runtime_data;
}