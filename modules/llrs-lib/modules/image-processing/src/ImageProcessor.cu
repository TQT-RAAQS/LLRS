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
 *     averaged PSF (area small enough so counts from neighboring PSFs don’t
 * overlap). Iterate through the psf size (the number of traps), for each trap
 * we get the weighted sum of it's pizels and add that to our current running
 * value. This step is known as "Deconvolution"
 */
void Processing::ImageProcessor::apply_filter(
    std::vector<uint16_t> &p_input_img, std::vector<int32_t> &current_config) {

    // Throw an error is we have an empty input image
    if (p_input_img.empty()) {
        throw std::runtime_error(
            "Image array not initialized to be processed."); // check image
                                                             // array
                                                             // initialization
    }
    // Initialize the return vector
    START_TIMER("II-Deconvolution");
    uint16_t *p_input_img_ptr = p_input_img.data();
    auto psf_ptr = this->_psf.data();
    size_t psf_size = this->_psf.size();
    auto running_sums_ptr = current_config.data();
// Iterate through all traps
#pragma omp parallel for firstprivate(p_input_img_ptr, psf_ptr, psf_size,      \
                                      running_sums_ptr) num_threads(16)
    for (size_t kernel_idx = 0; kernel_idx < psf_size; ++kernel_idx) {
        double cur_sum = 0;
        PSF_PAIR *p_psf = (psf_ptr + kernel_idx)->data();
        for (size_t i = 0; i < (psf_ptr + kernel_idx)->size(); ++i) {
            auto pair = *(p_psf + i);
            cur_sum +=
                *(p_input_img_ptr + std::get<0>(pair)) * std::get<1>(pair);
        }
#if IMAGE_INVERTED_X == true
        *(running_sums_ptr + psf_size - 1 - kernel_idx) =
            static_cast<int32_t>(cur_sum);
#else
        *(running_sums_ptr + kernel_idx) = static_cast<int32_t>(cur_sum);
#endif
    }
    END_TIMER("II-Deconvolution");
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

void Processing::ImageProcessor::apply_threshold(
    std::vector<int32_t> &filtered_vec, double threshold) {

    START_TIMER("II-Threshold");
    for (size_t trap_idx = 0; trap_idx < filtered_vec.size(); trap_idx++) {
        if (filtered_vec[trap_idx] >
            threshold) { // check if the trap contains an atom by comparing
                         // agaisnt the threshhold
            filtered_vec[trap_idx] = 1;
        } else {
            filtered_vec[trap_idx] = 0;
        }
    }
    END_TIMER("II-Threshold");
}

Processing::ImageProcessor::ImageProcessor() {
    this->reload();
}

size_t Processing::ImageProcessor::get_trap_count() {
    return this->psfs.size();
}

void Processing::ImageProcessor::process(
             size_t image_width,
             size_t image_index,
             const std::vector<uint16_t>& image,
             std::vector<double_t>& fls_counts,
             std::vector<uint8_t>& occupancy) {
    auto image_data = image.data();
    auto fls_data = fls_counts.data();
    auto occupancy_data = occupancy.data();
    auto psfs_data = this->psfs.data();
    auto thresholds_data = image_index >= this->thresholds.size() ? this->thresholds.back().data() : this->thresholds[image_index].data();

    auto trap_count = this->psfs.size();
    auto box_size = this->psfs[0].size();

    #pragma omp parallel for num_threads(16)
    for (size_t i = 0; i < trap_count; ++i) {
        double sum = 0;
        for (size_t j = 0; j < box_size; ++j) {
            auto& psf_tpl = *( ( *(psfs_data + i) ).data() + j );
            
            auto y = std::get<0>(psf_tpl);
            auto x = std::get<1>(psf_tpl);
            auto weight = std::get<2>(psf_tpl);

            sum += weight * *(image_data + (y*image_width) + x);
        }
        *(fls_data + i) = sum;
        *(occupancy_data + i) = (sum >= *(thresholds_data + i));
    }
}

void Processing::ImageProcessor::reload(bool flag_translate_psf) {
    if (flag_translate_psf) {
        this->configs_translator.translate_psf();
    }

    std::ifstream infile(PSF_TRANSLATION_FILE);
    this->parse_file(infile);
    infile.close();
}

void Processing::ImageProcessor::parse_file(std::ifstream& infile) {
    this->psfs.clear();
    this->thresholds.clear();

    // Read header
    int64_t trap_count, box_size_w, box_size_h, image_count;
    infile.read(reinterpret_cast<char*>(&trap_count), sizeof(trap_count));
    infile.read(reinterpret_cast<char*>(&box_size_w), sizeof(box_size_w));
    infile.read(reinterpret_cast<char*>(&box_size_h), sizeof(box_size_h));
    infile.read(reinterpret_cast<char*>(&image_count), sizeof(image_count));

    this->psfs.resize(trap_count);
    for (int64_t i = 0; i < trap_count; ++i) {
        int64_t yc, xc;
        infile.read(reinterpret_cast<char*>(&yc), sizeof(yc));
        infile.read(reinterpret_cast<char*>(&xc), sizeof(xc));

        auto& psf_vec = this->psfs[i];
        psf_vec.reserve(box_size_w * box_size_h);

        std::vector<double> psf_data(box_size_w * box_size_h);
        infile.read(reinterpret_cast<char*>(psf_data.data()), sizeof(double) * psf_data.size());

        for (int64_t j = 0; j < box_size_w; ++j) {
            for (int64_t k = 0; k < box_size_h; ++k) {
                int64_t y = yc - (box_size_w / 2) + k;
                int64_t x = xc - (box_size_h / 2) + j;
                double p = psf_data[j * box_size_h + k];
                psf_vec.emplace_back(y, x, p);
            }
        }
    }

    this->thresholds.resize(image_count);
    for (int64_t i = 0; i < image_count; ++i) {
        auto& thresholds_vec = this->thresholds[i];
        thresholds_vec.resize(trap_count);
        infile.read(reinterpret_cast<char*>(thresholds_vec.data()), sizeof(double) * trap_count);
    }
}