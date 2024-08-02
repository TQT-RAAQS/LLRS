#include "ImageProcessor.h"


/**
 *  @brief Write image data to a PGM file in binary format, with the filename
 * being the current epoch time in nanoseconds and saved in the
 * "output_data/images/" folder.
 *  @param image The image data represented as a vector of unsigned 16-bit
 * integers
 *  @param width The width of the image
 *  @param height The height of the image
 */
void Processing::write_to_pgm(const std::vector<uint16_t> &image, int width,
                              int height) {
    // Set the file name using the time of function call, then open the file to
    // write to
    std::string time_str =
        std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count());
    std::string file_name = IMAGE_PATH(time_str);

    std::ofstream ofs(file_name, std::ios::binary);

    if (!ofs) { // return error if unable to open file
        ERROR << "Could not open file " << file_name << " for writing"
              << std::endl;
        return;
    }

    // Write header
    const uint16_t max_value = 0xFFFF; // Max value for unsigned short
    ofs << "P2\n" << width << " " << height << "\n" << max_value << "\n";

    // Write pixel data to the file
    for (size_t i = 0; i < image.size(); i++) {
        ofs << image[i] << " ";
        if ((i + 1) % width == 0)
            ofs << "\n";
    }

    if (!ofs) { // print error if unable to write pixel data to the file
        ERROR << "Could not write pixel data to file " << file_name
              << std::endl;
    }
}

/**
 * @brief setup of ImageProcessor class, takes in path to psf file and
 * number of traps
 * @param psf_path the path to the psf file
 * @param num_traps number of traps
 */
void Processing::ImageProcessor::setup(std::string psf_path, size_t num_traps) {
    // Open the psf file in binary mode
    std::ifstream fin(psf_path, std::ios_base::in | std::ios_base::binary);

    // Throw exception if file not found
    if (!fin.is_open()) {
        throw std::invalid_argument("Psf file not found");
    }

    // Clear the existing _psf vector and resize it to the number of traps
    if (!this->_psf.empty()) {
        this->_psf.clear();
    }
    this->_psf.resize(num_traps);

    // Read array of (atom index, image index, corresponding _psf value) from
    // binary
    while (!fin.eof()) {
        // Read the kernel index, image index, and psf value
        size_t kernel_idx;
        size_t image_idx;
        double psf_value;
        fin.read((char *)&kernel_idx, sizeof(std::size_t));
        fin.read((char *)&image_idx, sizeof(std::size_t));
        fin.read((char *)&psf_value, sizeof(double));

        if (fin.eof()) {
            break;
        }

        // Ignore entries with kernel index greater than or equal to num_traps
        if (kernel_idx >= num_traps) {
            continue;
        }

        // Add the (image index, psf value) pair to the corresponding kernel
        // index vector
        this->_psf.at(kernel_idx)
            .push_back(std::make_pair(image_idx, psf_value));
    }

    // Close the file
    fin.close();
}

/**
 *   @brief Applies a filter to the image using the stored PSF and returns an
 * array indicating the presence or absence of atoms in the corresponding traps.
 *   @param p_input_img Input image taken directly from the camera
 *   @return An array representing whether an atom is detected in each trap of
 * the image. e.g. [1, 0, 1] => [atom detected, no atom, atom detected]
 *   @throws std::runtime_error if the image array is not initialized.
 *   Processing Steps:
 *     Determine the location of the atom's point-spread function (PSF) in the
 * image by averaging several images and finding the centroid. Derive a
 * numerical signal from the image, such as the weighted sum over pixels within
 * the PSF area, where the weights are obtained from the pixel values of the
 *       averaged PSF (area small enough so counts from neighboring PSFs don’t
 * overlap). Iterate through the psf size (the number of traps), for each trap
 * we get the weighted sum of it's pizels and add that to our current running
 * value. This step is known as "Deconvolution"
 */
std::vector<double>
Processing::ImageProcessor::apply_filter(std::vector<uint16_t> *p_input_img) {

    START_TIMER("II-Deconvolution");
    // Throw an error is we have an empty input image
    if (p_input_img->empty()) {
        throw std::runtime_error(
            "Image array not initialized to be processed."); // check image
                          // array
                          // initialization
    }
    // Initialize the return vector
    std::vector<double> running_sums(this->_psf.size(), 0);
    uint16_t* p_input_img_ptr = p_input_img->data();
    auto psf_ptr = this->_psf.data();
 
    // Iterate through all traps
    #pragma omp parallel for firstprivate(p_input_img_ptr, psf_ptr) num_threads(FILTERING_NUM_THREADS) 
    for (size_t kernel_idx = 0; kernel_idx < _psf.size(); kernel_idx++) {
        double cur_sum = 0;
        PSF_PAIR* p_psf = (psf_ptr + kernel_idx)->data();
        for (size_t i = 0; i < _psf[kernel_idx].size(); i++) {
            auto pair = *(p_psf + i);
            cur_sum += *(p_input_img_ptr+std::get<0>(pair)) * std::get<1>(pair);
        }
#if IMAGE_INVERTED_X == true
        running_sums[this->_psf.size() - 1 - kernel_idx] = cur_sum;
#else
        running_sums[kernel_idx] = cur_sum;
#endif
    }
    END_TIMER("II-Deconvolution");

    return std::move(running_sums);
}

/**
 * Apply a threshold to the filtered vector to determine which traps contain
 * atoms.
 * @param filtered_vec The vector of filtered values to threshold.
 * @param threshold The threshold value to use for classification.
 * @return A vector of integers representing whether each trap contains an atom
 * (1) or not (0). Itarate through all traps and determine if the contain an
 * atom (if the value is above a certain threshold). This step is known as
 * Thresholding.
 */

std::vector<int32_t>
Processing::ImageProcessor::apply_threshold(std::vector<double> filtered_vec,
                                            double threshold) {

    START_TIMER("II-Threshold");
    std::vector<int32_t> atom_configuration(filtered_vec.size(), 0);
    for (size_t trap_idx = 0; trap_idx < filtered_vec.size(); trap_idx++) {
        if (filtered_vec[trap_idx] >
            threshold) { // check if the trap contains an atom by comparing
                         // agaisnt the threshhold
            atom_configuration[trap_idx] = 1;
        }
    }
    END_TIMER("II-Threshold");

    return std::move(atom_configuration);
}
