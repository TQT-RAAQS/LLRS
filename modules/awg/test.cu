#include "awg.hpp"
#include "log.h"
#include "math.h"

#define M 32767

int main() {
    auto awg = AWG{"iqmixer.yml"};
    int N = 62'400'00;
    std::cout << "Segment size: " << N << std::endl;

    std::vector<short> vI(N), vQ(N);
    std::vector<int8> vD(N);
    double freq = 100e6, alpha = 0.01;
    int sr = awg.get_sample_rate();

    auto analog_func = [freq, alpha, sr](int channel, int ind) {
        auto t = static_cast<double>(ind) / sr;
        return static_cast<short>( (channel == 0 ? alpha : alpha) * M * sin(2 * M_PI * freq * t) );
    };
    auto digital_func = [sr](int channel, int ind) {
        return static_cast<uint8>(1);
    };

    awg.open_connection();
    auto buffer1 = awg.allocate_transfer_buffer(N, false);
    auto buffer2 = awg.allocate_transfer_buffer(N, false);
    auto buffer3 = awg.allocate_transfer_buffer(N, false);

    // Method 1
    auto before = std::chrono::high_resolution_clock::now();
    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        auto t = static_cast<double>(i) / awg.get_sample_rate();
        vI[i] = static_cast<short>(M * alpha * sin(2 * M_PI * freq * t));
        vQ[i] = static_cast<short>(M * alpha * sin(2 * M_PI * freq * t));
        vD[i] = 1;
    }
    awg.interleave_data(*buffer1, {vI, vQ}, {vD});
    auto after = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << " ms (method 1)" << std::endl;

    // Method 2
    before = std::chrono::high_resolution_clock::now();
    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        auto t = static_cast<double>(i) / awg.get_sample_rate();
        (*buffer2)[2 * i] = static_cast<short> ( (static_cast<uint16>(M * alpha * sin(2 * M_PI * freq * t)) >> 1) | 0x8000 );
        (*buffer2)[2 * i + 1] = static_cast<short>( M * alpha * sin(2 * M_PI * freq * t) );
    }
    after = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << " ms (method 2)" << std::endl;

    // Method 3
    before = std::chrono::high_resolution_clock::now();
    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < 2; ++c) {
            (*buffer3)[2 * i + c] = static_cast<short>( (c == 0 ? ( (static_cast<uint16>(analog_func(c, i)) >> 1) | (0x8000) ) : analog_func(c, i)) );
        }
    }
    after = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << " ms (method 3)" << std::endl;

    // Verification for method 2 and 3
    for (int i = 0; i < 2*N; ++i) {
        assert ((*buffer1)[i] == (*buffer2)[i]);
        assert ((*buffer1)[i] == (*buffer3)[i]);
    }

    awg.close_card();

    return 0;
}