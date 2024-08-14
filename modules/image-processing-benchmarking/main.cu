#include "ImageProcessor.h"
#include <chrono>
#include <cmath>
#include <ctime>
#include <numeric>

int reps = 100;
int N_t = 32;
int kernel_size = 1;
std::string psf_path{};

// function to take in command line args for height and psf_path, width stays
// constant
int cmd_line(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Usage is: ./image_processing_unit <psf_path> <Nt> "
                     "<kernel_size> <reps>"
                  << std::endl;
        return 1;
    }
    psf_path = std::string(argv[1]);
    N_t = std::stoi(argv[2]);
    kernel_size = std::stoi(argv[3]);
    reps = std::stoi(argv[4]);
    return 0;
}

int main(int argc, char *argv[]) {

    // taking in command line arguments
    if (cmd_line(argc, argv) != 0) {
        return 1;
    }

    // creating vector to store all times
    std::vector<double> results;
    results.reserve(reps);

    // initialize Image Processing object
    Processing::ImageProcessor img_proc_obj{};
    img_proc_obj.setup(psf_path, N_t);
    // initialize vector for current image
    std::vector<uint16_t> current_image(MAX_IMAGE_READ_BUFFER_SIZE, 0);
    std::vector<int32_t> current_config(N_t, 0);

    // arbitrarily change the values for the vector of pixels
    for (int j = 0; j < MAX_IMAGE_READ_BUFFER_SIZE; ++j) {
        current_image[j] = j % 16;
    }
    for (int i = 0; i < reps; ++i) {
        std::chrono::steady_clock::time_point begin;
        std::chrono::steady_clock::time_point end;

        begin = std::chrono::steady_clock::now();
        img_proc_obj.apply_filter(current_image, current_config);
        end = std::chrono::steady_clock::now();

        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      begin);
        std::cerr << current_config.size();
        results.push_back(time_span.count());
    }

    double average =
        std::accumulate(results.begin(), results.end(), 0.0) / reps;
    std::vector<double> diffs;
    diffs.reserve(results.size());
    for (auto it : results) {
        diffs.push_back(it - average);
    }
    double stddev = std::sqrt(
        std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
        (reps - 1));
    std::cout << average << std::endl << stddev << std::endl;
}
