#include "Collector.h"

Util::Collector *Util::Collector::p_instance{nullptr};

Util::Collector *Util::Collector::get_instance() {
    if (p_instance == nullptr) {
        p_instance = new Collector();
        p_instance->_timers.reserve(7);
    }
    return p_instance;
}

void Util::Collector::get_external_time(const std::string &module, int trial,
                                        int rep, int cycle, float time) {

    if (_timers.find(module) == _timers.end()) {
        _timers[module] = TRIAL_INFO();
    }
    while (_timers[module].size() <= trial) {
        _timers[module].push_back(REP_INFO());
    }
    while (_timers[module][trial].size() <= rep) {
        _timers[module][trial].push_back(CYCLE_INFO());
    }
    while (_timers[module][trial][rep].size() <= cycle) {
        _timers[module][trial][rep].push_back(TIMING_PAIR());
    }

    std::chrono::nanoseconds external_time{static_cast<int>(round(time))};

    auto time_now = std::chrono::high_resolution_clock::now();

    auto time_now_plus_alg_elapse = time_now + external_time;

    _timers[module][trial][rep][cycle] =
        std::make_pair(time_now, time_now_plus_alg_elapse);
}

void Util::Collector::start_timer(const std::string &module, int trial, int rep,
                                  int cycle) {
    if (_timers.find(module) == _timers.end()) {
        _timers[module] = TRIAL_INFO();
    }
    while (_timers[module].size() <= trial) {
        _timers[module].push_back(REP_INFO());
    }
    while (_timers[module][trial].size() <= rep) {
        _timers[module][trial].push_back(CYCLE_INFO());
    }
    while (_timers[module][trial][rep].size() <= cycle) {
        _timers[module][trial][rep].push_back(TIMING_PAIR());
    }

    auto time_now = std::chrono::high_resolution_clock::now();
    _timers[module][trial][rep][cycle] = std::make_pair(time_now, time_now);
}

void Util::Collector::end_timer(const std::string &module, int trial, int rep,
                                int cycle) {
    auto finish = std::chrono::high_resolution_clock::now();
    _timers[module][trial][rep][cycle].second = finish;
}

long long Util::Collector::elapsed_time(const std::string &module, int trial,
                                        int rep, int cycle) {
    if (_timers.find(module) == _timers.end()) {
        return 0;
    }
    auto cycle_pair = _timers[module][trial][rep][cycle];
    auto start_time = cycle_pair.first;
    auto end_time = cycle_pair.second;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                start_time)
        .count();
}

nlohmann::json Util::Collector::gen_runtime_json() {
    nlohmann::json root;

    for (const auto &module_info : _timers) {
        const std::string &module = module_info.first;
        const TRIAL_INFO &trial_info = module_info.second;
        for (int i = 0; i < trial_info.size(); ++i) {
            std::string trial_name = TRIAL_NAME(i);
            const REP_INFO &rep_info = trial_info.at(i);
            for (int j = 0; j < rep_info.size(); ++j) {
                std::string repetition_name = REP_NAME(j);
                const CYCLE_INFO &cycle_info = rep_info.at(j);
                for (int k = 0; k < cycle_info.size(); k++) {
                    std::string cycle_name = CYCLE_NAME(k);
                    root[trial_name][repetition_name][cycle_name][module] =
                        elapsed_time(module, i, j, k);
                }
            }
        }
    }

    return root;
}