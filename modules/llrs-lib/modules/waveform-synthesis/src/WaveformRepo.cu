#include "WaveformRepo.h"

/// Helper Functions

inline std::uint32_t Synthesis::get_key(Channel channel, WfType type,
                                        std::uint32_t index,
                                        std::uint32_t block_size,
                                        std::uint32_t extraction_extent) {

    return (channel << 30) + (type << 26) + (index << 18) + (block_size << 10) +
           extraction_extent;
}

void Synthesis::element_wise_add(std::vector<double> src,
                                 std::vector<double> &dst) {
    assert(src.size() == dst.size());

    for (int i = 0; i < dst.size(); i++) {
        dst[i] = dst[i] + src[i];
    }
}

/// WaveformRepo Class

Synthesis::WaveformRepo::WaveformRepo(std::size_t n_x, std::size_t n_y,
                                      double sample_rate, double wf_duration)
    : _n_x(n_x), _n_y(n_y), _sample_rate(sample_rate),
      _wf_duration(wf_duration) {
    _wf_len = (size_t)(_wf_duration * _sample_rate);
    // reserves enough space in repo to fit all waveforms according to repo
    // dimensions
    _waveform_hashmap.reserve((5 * _n_x * _n_x + _n_x) / 2 +
                              (5 * _n_y * _n_y + _n_y) / 2 +
                              2 * (_n_x + _n_y - 2));
}

void Synthesis::WaveformRepo::generate_repository(std::vector<WP> params_x,
                                                  std::vector<WP> params_y,
                                                  double df) {

    // generate waveforms for x axis
    generate_static_waveforms(CHAN_0, params_x, _n_x);
    generate_displacement_waveforms(CHAN_0, params_x, _n_x, df);
    generate_transfer_waveforms(CHAN_0, params_x, _n_x);

    // generate waveforms for y axis
    generate_static_waveforms(CHAN_1, params_y, _n_y);
    generate_displacement_waveforms(CHAN_1, params_y, _n_y, df);
    generate_transfer_waveforms(CHAN_1, params_y, _n_y);
}

std::vector<double> Synthesis::WaveformRepo::get_waveform(
    Channel channel, WfType waveform_type, std::uint32_t index,
    std::uint32_t block_size, std::uint32_t extraction_extent) const {
    int key =
        get_key(channel, waveform_type, index, block_size, extraction_extent);
    return _waveform_hashmap.at(key);
}

std::vector<double>
Synthesis::WaveformRepo::get_waveform(std::uint32_t key) const {
    return _waveform_hashmap.at(key);
}

int Synthesis::WaveformRepo::cache_waveform_repo(std::string fname) const {
    std::ofstream file(fname, std::ios::out | std::ios::binary);

    if (!file.is_open())
        return LLRS_ERR;

#ifdef POWER_SAFETY
    WaveformPowerCalculator wpc;
#endif
    for (const auto &data : _waveform_hashmap) {

        file.write((char *)&data.first, sizeof(uint32_t));

        for (const auto &val : data.second) {
            file.write((char *)&val, sizeof(double));
        }

#ifdef POWER_SAFETY
        if (!wpc.is_power_safe(data.second)) {
            file.close();
            std::remove(fname.c_str());
            std::cerr << "Power exceeded allowed threshold. Power = "
                      << wpc.get_power_dBm(data.second)
                      << "; Key=" << data.first << std::endl;
            throw std::runtime_error("Power exceeded the allowed threshold for "
                                     "a waveform when caching.");
        }
#endif
    }

    file.close();

    return LLRS_OK;
}

int Synthesis::WaveformRepo::load_waveform_repo(std::string fname) {
    std::ifstream file(fname, std::ios::in | std::ios::binary);
#ifdef POWER_SAFETY
    WaveformPowerCalculator wpc;
#endif

    if (!file.is_open())
        return LLRS_ERR;

    uint32_t key;

    std::vector<double> wf(_wf_len);

    while (file.read((char *)&key, sizeof(uint32_t))) {

        for (int i = 0; i < _wf_len; i++) {
            file.read((char *)&wf[i], sizeof(double));
        }

#ifdef POWER_SAFETY
        if (!wpc.is_power_safe(wf)) {
            file.close();
            std::cerr << "Power exceeded allowed threshold. Power = "
                      << wpc.get_power_dBm(wf) << "; Key=" << key << std::endl;
            throw std::runtime_error("Power exceeded the allowed threshold for "
                                     "a waveform when loading.");
        }
#endif
        _waveform_hashmap[key] = wf;
    }

    file.close();

    return LLRS_OK;
}

int Synthesis::WaveformRepo::cache_waveform_repo_plaintext(
    std::string fname) const {
    std::ofstream file(fname);

    if (!file.is_open())
        return LLRS_ERR;

    for (const auto &data : _waveform_hashmap) {

        file << data.first << " " << (data.second).size();

        for (const auto &val : data.second) {
            file << " " << val;
        }

        file << std::endl;
    }

    file.close();

    return LLRS_OK;
}

int Synthesis::WaveformRepo::load_waveform_repo_plaintext(std::string fname) {
    std::ifstream file(fname);

    if (!file.is_open())
        return LLRS_ERR;

    std::string line;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> v;
        int id;
        size_t sz;

        ss >> id >> sz;
        v.reserve(sz);
        double a;
        while (ss >> a) {
            v.push_back(a);
        }
        _waveform_hashmap.emplace(id, std::move(v));
    }

    file.close();
    return LLRS_OK;
}

std::unordered_map<std::uint32_t, std::vector<double>>
Synthesis::WaveformRepo::get_raw_map() const {
    return this->_waveform_hashmap;
}

std::int32_t Synthesis::WaveformRepo::get_repo_size() {
    return _waveform_hashmap.size();
}

std::size_t Synthesis::WaveformRepo::get_dimension_x() { return this->_n_x; }
std::size_t Synthesis::WaveformRepo::get_dimension_y() { return this->_n_y; }

bool Synthesis::WaveformRepo::operator==(const WaveformRepo &other) const {
    return (_n_x == other._n_x) && (_n_y == other._n_y) &&
           (_waveform_hashmap == other._waveform_hashmap);
}

bool Synthesis::WaveformRepo::operator!=(const WaveformRepo &other) const {
    return !(*this == other);
}

void Synthesis::WaveformRepo::generate_static_waveforms(Channel channel,
                                                        std::vector<WP> params,
                                                        std::size_t n) {

    /// used to create block waveforms
    int end_idx;

    /// vector to temporarily store single tone waveforms
    std::vector<std::vector<double>> temp_waveforms(n);

    /// generate every indivisual tone first and add it to temp vector and
    /// hashmap
    for (int i = 0; i < n; ++i) {
        Waveform wf(params[i]);
        std::vector<double> dwf = wf.discretize(_wf_len, _sample_rate);

        _waveform_hashmap.emplace(get_key(channel, STATIC, i, 1, 0), dwf);
        temp_waveforms[i] = std::move(dwf); // move prevents copies
    }

    /// use individual tones to generate block waveforms starting at start_idx
    /// and ending at end_idx

    for (int start_idx = 0; start_idx < n; start_idx++) {

        /// we minimize summations by only keeping track of one sum and copying
        /// it to the map as we go
        std::vector<double> base = temp_waveforms[start_idx];

        for (int block_size = 2; start_idx + block_size - 1 < n; block_size++) {

            end_idx = start_idx + block_size - 1;

            /// add single tone to sum
            element_wise_add(temp_waveforms[end_idx], base);

            /// put copy of sum into repo
            _waveform_hashmap.emplace(
                get_key(channel, STATIC, start_idx, block_size, 0), base);
        }
    }
}

void Synthesis::WaveformRepo::generate_displacement_waveforms(
    Channel channel, std::vector<WP> params, std::size_t n, double df) {

    // used for backward waveforms
    int reverse_i;
    int reverse_sidx;

    // used for block waveforms
    int end_idx;
    int reverse_eidx;

    // vectors to temorarily store pieces of block displacement waveforms
    std::vector<std::vector<double>> fade_in_forward(n);
    std::vector<std::vector<double>> fade_in_backward(n);
    std::vector<std::vector<double>> fade_out_forward(n);
    std::vector<std::vector<double>> fade_out_backward(n);
    std::vector<std::vector<double>> move_forward(n);
    std::vector<std::vector<double>> move_backward(n);

    // generate individual fading displacement waveforms
    for (int i = 0; i < n; i++) {

        WP params_before(0, std::get<1>(params[i]) - df, 0);
        WP params_after(0, std::get<1>(params[i]) + df, 0);

        fade_in_forward[i] = Displacement(params_before, params[i])
                                 .discretize(_wf_len, _sample_rate);
        fade_in_backward[i] = Displacement(params_after, params[i])
                                  .discretize(_wf_len, _sample_rate);

        fade_out_forward[i] = Displacement(params[i], params_after)
                                  .discretize(_wf_len, _sample_rate);
        fade_out_backward[i] = Displacement(params[i], params_before)
                                   .discretize(_wf_len, _sample_rate);
    }

    // generate individual moving displacement waveforms, add them to repo as
    // special single tone displacements
    for (int i = 0; i < n - 1; i++) {
        reverse_i = n - 1 - i;
        move_forward[i] = Displacement(params[i], params[i + 1])
                              .discretize(_wf_len, _sample_rate);
        move_backward[reverse_i] =
            Displacement(params[reverse_i], params[reverse_i - 1])
                .discretize(_wf_len, _sample_rate);

        _waveform_hashmap.emplace(get_key(channel, RIGHTWARD, i, 1, 0),
                                  move_forward[i]);
        _waveform_hashmap.emplace(
            get_key(channel, LEFTWARD, reverse_i - 1, 1, 0),
            move_backward[reverse_i]);
    }

    // block waveform generation

    // temps are needed to avoid adding edge fading waveforms to intermediary
    // sums
    std::vector<double> block_base_forward;
    std::vector<double> block_base_backward;
    std::vector<double> temp_forward;
    std::vector<double> temp_backward;

    // constructing and storing block displacement waveforms
    for (int start_idx = 0; start_idx < n - 1; start_idx++) {

        reverse_sidx = n - 1 - start_idx;

        // base wavforms have fade at start added to them
        std::vector<double> base_forward = fade_in_forward[start_idx];
        std::vector<double> base_backward = fade_in_backward[reverse_sidx];

        // minimizing sums as in transfer and static
        for (int block_size = 1; start_idx + block_size - 1 < n - 1;
             block_size++) {
            end_idx = start_idx + block_size - 1;
            reverse_eidx = reverse_sidx - block_size + 1;

            // adding to base sum
            element_wise_add(move_forward[end_idx], base_forward);
            element_wise_add(move_backward[reverse_eidx], base_backward);

            block_base_forward = fade_out_forward[end_idx + 1];
            block_base_backward = fade_out_backward[reverse_eidx - 1];

            // adding fading at end, without touching the base sum
            element_wise_add(base_forward, block_base_forward);
            element_wise_add(base_backward, block_base_backward);

            for (int extraction_extent = start_idx + block_size + 1;
                 extraction_extent <= n; extraction_extent++) {
                // add auxilliary statics on either side, without touching the
                // base sum for the blocksize context
                temp_forward = block_base_forward;
                temp_backward = block_base_backward;

                // before disp. block
                if (start_idx > 0) {

                    // forward
                    element_wise_add(
                        get_waveform(channel, STATIC, 0, start_idx, 0),
                        temp_forward);
                    // backward
                    element_wise_add(get_waveform(channel, STATIC,
                                                  reverse_sidx + 1,
                                                  n - (reverse_sidx + 1), 0),
                                     temp_backward);
                }

                // after disp. block
                if (end_idx < extraction_extent - 2) {
                    // forward
                    element_wise_add(
                        get_waveform(channel, STATIC, end_idx + 2,
                                     extraction_extent - (end_idx + 2), 0),
                        temp_forward);
                    // backward
                    element_wise_add(get_waveform(channel, STATIC,
                                                  n - extraction_extent,
                                                  1 + (reverse_eidx - 2) -
                                                      (n - extraction_extent),
                                                  0),
                                     temp_backward);
                }

                // storing
                _waveform_hashmap.emplace(get_key(channel, FORWARD, start_idx,
                                                  block_size,
                                                  extraction_extent),
                                          std::move(temp_forward));

                // std::cout << reverse_eidx-1 << " " << block_size << " " <<
                // extraction_extent << std::endl;
                _waveform_hashmap.emplace(get_key(channel, BACKWARD,
                                                  reverse_eidx - 1, block_size,
                                                  extraction_extent),
                                          std::move(temp_backward));
            }
        }
    }
}

void Synthesis::WaveformRepo::generate_transfer_waveforms(
    Channel channel, std::vector<WP> params, std::size_t n) {

    /// used for block waveforms
    int end_idx;

    /// used to temporarily store single tone waveforms
    std::vector<std::vector<double>> temp_waveforms_extract(n);
    std::vector<std::vector<double>> temp_waveforms_implant(n);

    /// generating and storing individual tone waveforms
    for (size_t i = 0; i < n; ++i) {
        Extraction ewf(params[i], false);
        std::vector<double> edwf = ewf.discretize(_wf_len, _sample_rate);

        Extraction iwf(params[i], true);
        std::vector<double> idwf = iwf.discretize(_wf_len, _sample_rate);

        _waveform_hashmap.emplace(get_key(channel, EXTRACT, i, 1, 0), edwf);
        temp_waveforms_extract[i] = std::move(edwf);

        _waveform_hashmap.emplace(get_key(channel, IMPLANT, i, 1, 0), idwf);
        temp_waveforms_implant[i] = std::move(idwf);
    }

    /// constructing and storing block waveforms
    for (int start_idx = 0; start_idx < n; start_idx++) {

        /// we minimize summations by only keeping track of one sum and copying
        /// it to the map as we go
        std::vector<double> base_extract = temp_waveforms_extract[start_idx];
        std::vector<double> base_implant = temp_waveforms_implant[start_idx];

        for (int block_size = 2; start_idx + block_size - 1 < n; block_size++) {

            end_idx = start_idx + block_size - 1;

            element_wise_add(temp_waveforms_extract[end_idx], base_extract);
            element_wise_add(temp_waveforms_implant[end_idx], base_implant);

            _waveform_hashmap.emplace(
                get_key(channel, EXTRACT, start_idx, block_size, 0),
                base_extract);
            _waveform_hashmap.emplace(
                get_key(channel, IMPLANT, start_idx, block_size, 0),
                base_implant);
        }
    }
}
