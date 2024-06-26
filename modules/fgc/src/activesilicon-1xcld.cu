/*
 * Instrument class for the activesilicon_1xcld
 * Author: Laurent Zheng
 * Winter 2023
 */

#include "activesilicon-1xcld.hpp"
#include <png.h>

/**
 *
 *   @brief Setup for ActiveSilicon1XCLD acquisition class.
 *   @param roi_width The width of the region of interest (ROI) in pixels.
 *   @param roi_height The height of the region of interest (ROI) in pixels.
 *   @param roi_xoffset The x offset of the top-left corner of the ROI from the
 * top-left corner of the camera sensor.
 *   @param roi_yoffset The y offset of the top-left corner of the ROI from the
 * top-left corner of the camera sensor.
 */
void Acquisition::ActiveSilicon1XCLD::setup(
    uint32_t roi_width, uint32_t roi_height, int acquisition_timeout,
    uint32_t roi_xoffset, uint32_t roi_yoffset, uint32_t vbin, uint32_t hbin) {
    _roi_width = roi_width;
    _roi_height = roi_height;
    _roi_xoffset = roi_xoffset;
    _roi_yoffset = roi_yoffset;
    _vbin = vbin;
    _hbin = hbin;
    _acquisition_timeout = acquisition_timeout;
    /* Create handle and load default settings base on the PCF file */
    std::string NADA = "";
    std::string pcf_config_path = std::string(
        NADA + PROJECT_BASE_DIR + "/configs/fgc/andor_ixonultra888.pcf");
    PHX_Create(&this->_handle, &PHX_ErrHandlerDefault);
    PHX_ParameterSet(this->_handle, PHX_CONFIG_FILE, &pcf_config_path);

    /* Static Configurations Customize */
    int32_t phx_board1 = (int32_t)PHX_BOARD1;
    PHX_ParameterSet(this->_handle, PHX_BOARD_NUMBER, &phx_board1);
    int32_t phx_channel_number_1 = (int32_t)PHX_CHANNEL_NUMBER_1;
    PHX_ParameterSet(this->_handle, PHX_CHANNEL_NUMBER, &phx_channel_number_1);
    int32_t phx_cameraconfig_load = (int32_t)PHX_CAMERACONFIG_LOAD;
    PHX_ParameterSet(this->_handle, PHX_CONFIG_MODE, &phx_cameraconfig_load);
    PHX_Open(this->_handle);

    // /* Dynamic Configurations Customize: Specify camera active region*/
    PHX_ParameterSet(this->_handle, PHX_CAM_ACTIVE_XOFFSET,
                     &this->_roi_xoffset);
    PHX_ParameterSet(this->_handle, PHX_CAM_ACTIVE_YOFFSET,
                     &this->_roi_yoffset);
    PHX_ParameterSet(this->_handle, PHX_CAM_ACTIVE_XLENGTH, &this->_roi_width);
    PHX_ParameterSet(this->_handle, PHX_CAM_ACTIVE_YLENGTH, &this->_roi_height);

    /* Note: the region of interest is the same as the active camera region */
    PHX_ParameterSet(this->_handle, PHX_ROI_XLENGTH, &this->_roi_width);
    PHX_ParameterSet(this->_handle, PHX_ROI_YLENGTH, &this->_roi_height);

    /* Configure binning */
    PHX_ParameterSet(this->_handle, PHX_CAM_XBINNING, &this->_hbin);
    PHX_ParameterSet(this->_handle, PHX_CAM_YBINNING, &this->_vbin);

    /* Force a flush of all parameters writes to hardware */
    PHX_ParameterSet(this->_handle,
                     (etParam)(PHX_DUMMY_PARAM | PHX_CACHE_FLUSH), NULL);
    init_image_buffer();
}

/**
 *   @brief Initializes the image buffer for ActiveSilicon1XCLD acquisition
 * class. This function initializes the image buffer for ActiveSilicon1XCLD
 * acquisition class. If CUDA is enabled, it allocates memory on the GPU and
 * creates a GPUDirect buffer, otherwise it allocates memory on the CPU.
 */
void Acquisition::ActiveSilicon1XCLD::init_image_buffer() {
#ifdef ENABLE_CUDA
    cudaMalloc((void **)&gpu_buffers_src,
               MAX_IMAGE_READ_BUFFER_SIZE * sizeof(uint16_t));
    /* gpu_buffer_desc-> a class stores information for the buffer not the
       buffer it self create a GPUDirectBuffer use to create and allocate the
       buffer  */
    gpuObjectDesc gpu_buffer_desc = {
        False,
        1024,
        1024,
        1,
        2,
        0,
        MAX_IMAGE_READ_BUFFER_SIZE * sizeof(uint16_t)};
    /* Initialise this object with the GPU object provided (allocates system
     * memory for DMA, semaphore, etc.) */
    gpu_buffer.Init(gpu_buffer_desc, (void **)&gpu_buffers_src, 16);
    /* image buffer stores the address of the gpu buffer */
    this->_image_buffers[0].pvAddress = gpu_buffer.GetSystemMemory();
    this->_image_buffers[1].pvAddress = NULL;
#else
    this->_image_buffers[0].pvAddress =
        new uint16_t[MAX_IMAGE_READ_BUFFER_SIZE];
    this->_image_buffers[1].pvAddress = NULL;
#endif
    // Sets virtual memory buffers for the frame grabber
    etParamValue phx_dst_ptr_user_virt = (etParamValue)PHX_DST_PTR_USER_VIRT;
    PHX_ParameterSet(this->_handle, PHX_DST_PTR_TYPE, &phx_dst_ptr_user_virt);
    PHX_ParameterSet(this->_handle, PHX_DST_PTRS_VIRT, this->_image_buffers);
}

std::vector<uint16_t>
Acquisition::ActiveSilicon1XCLD::acquire_stored_image(const char *filename) {

    FILE *file = fopen(filename, "rb");
    if (!file) {
        std::cerr << "Error: Cannot open file." << std::endl;
        return {};
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                                 nullptr, nullptr);
    if (!png_ptr) {
        std::cerr << "Error: Cannot create PNG read structure." << std::endl;
        fclose(file);
        return {};
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Error: Cannot create PNG info structure." << std::endl;
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        fclose(file);
        return {};
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Error during PNG creation." << std::endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(file);
        return {};
    }

    png_init_io(png_ptr, file);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, nullptr);

    png_bytep *row_pointers = png_get_rows(png_ptr, info_ptr);
    png_uint_32 width = png_get_image_width(png_ptr, info_ptr);
    png_uint_32 height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Verify that the image is 16-bit depth per channel
    if (bit_depth != 16) {
        std::cerr << "Error: Image is not 16-bit per channel." << std::endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(file);
        return {};
    }

    std::vector<uint16_t> image_data;
    image_data.reserve(width * height);

    for (png_uint_32 y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (png_uint_32 x = 0; x < (width * 2); x += 2) {
            // Combine two bytes into one uint16_t value
            uint16_t value = (static_cast<uint16_t>(row[x]) << 8) | row[x + 1];
            image_data.push_back(value);
        }
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(file);

    return std::move(image_data);
}

/**
 *   @brief Acquire a single image from the camera.
 *   @return A vector of uint16_t integers containing the pixel values of the
 * acquired image. If an error occurs or the acquisition times out, an empty
 * vector is returned.
 */
std::vector<uint16_t> Acquisition::ActiveSilicon1XCLD::acquire_single_image() {
    start_stream_read(); // begin the stream read

    std::vector<uint16_t> ret_image;
    int status = check_and_wait();
    if (status != 0) {
        ret_image = get_current_buffer(); // get the current buffer contents
                                          // when check_and_wait finishes
        stop_stream_read();               // stop the stream read
        std::cout << "Image Acquisition Successful." << std::endl;

    } else {
        std::cerr
            << "WARNING: timed out on image acquistion."
            << std::endl; // check_and_wait did not finish in a suitable time
    }

    release_current_buffer();
    return std::move(ret_image);
}

/**
 *   @brief Retrieves the current buffer of acquired image data and transfers it
 * to the host.
 *   @return A vector containing the acquired image data.
 */
std::vector<uint16_t> Acquisition::ActiveSilicon1XCLD::get_current_buffer() {
    PHX_StreamRead(this->_handle, PHX_BUFFER_GET, this->_image_buffers);

    std::vector<uint16_t> ret_image(this->_roi_width * this->_roi_height, 0);
#ifdef ENABLE_CUDA
    gpu_buffer.TransferChunk();
    gpu_buffer.GetFrame(); // sets up the rdma transfer from fgc to gpu blocks
    gpu_buffer.TransferChunk(); // triggers the rdma transfer that is set up by
                                // get_frame
    cudaMemcpy(&ret_image[0], gpu_buffers_src,
               this->_roi_width * this->_roi_height * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    gpu_buffer.EndFrame();
#else
    memcpy(&ret_image[0], this->_image_buffers[0].pvAddress,
           this->_roi_width * this->_roi_height * sizeof(uint16_t));
#endif

    return std::move(ret_image);
}

/**
 *   @brief Checks and waits for an event from the frame grabber, using a
 * timeout. an event variable will be set to the value of the event that
 * occurred.
 *   @return true if an event occurred, false otherwise.
 */
int32_t Acquisition::ActiveSilicon1XCLD::check_and_wait() {
    int32_t event = 0;
    boost::asio::io_service io_service;
    boost::asio::deadline_timer timer(
        io_service, boost::posix_time::milliseconds(_acquisition_timeout));

    std::thread event_thread([&] {
        PHX_StreamRead(this->_handle, PHX_CHECK_AND_WAIT, &event);
        timer.cancel();
        io_service.stop();
    });

    timer.async_wait(
        [&](const boost::system::error_code &error) { // asynchronous timer
            if (!error && (event == 0)) { // if no event occured and no error
                                          // then abort the stream read
                abort_stream_read();
            }
            io_service.stop(); // Determine whether the io_service object has
                               // been stopped
        });

    io_service.run(); // Run the io_service's event processing loop.

    if (event == 0) { // if no event then detach the thread
        event_thread.detach();
    } else { // otherwise join the thread
        event_thread.join();
    }

    return event; // return whether or not an event occurred.
}

/**
 *   @brief Destructor for ActiveSilicon1XCLD acquisition class.
 *   This function deallocates any memory or resources that were allocated
 * during the lifetime
 */
Acquisition::ActiveSilicon1XCLD::~ActiveSilicon1XCLD() {
#ifdef ENABLE_CUDA
    gpu_buffer.DeInit();
    gpu_buffer.CloseDevice();
    cudaFree(gpu_buffers_src);
#endif

    if (this->_image_buffers[0].pvAddress != NULL) {
        delete[] this->_image_buffers[0].pvAddress;
    }

    // delete[] static_cast<uint16_t*>(this->_image_buffers[0].pvAddress) ;
    delete[] this->_image_buffers;
}

/**
 *   @brief Destroys the handle for ActiveSilicon1XCLD acquisition class.
 *
 */
bool Acquisition::ActiveSilicon1XCLD::destroy_handle() {
    if (this->_handle) {
        if (PHX_Destroy(&this->_handle) !=
            PHX_OK) {          // PHX_Destroy destroys the handle
            return CAMERA_ERR; // return LLRS_ERR if destroying the handle does
                               // not occur
        }
    }
    return CAMERA_OK;
}

// Repackaging of driver calls from the PHX library for the Active Silicon Card
// indicated by _handle
void Acquisition::ActiveSilicon1XCLD::start_stream_read() {
    PHX_StreamRead(this->_handle, PHX_START, NULL);
}
void Acquisition::ActiveSilicon1XCLD::release_current_buffer() {
    PHX_StreamRead(this->_handle, PHX_BUFFER_RELEASE, NULL);
}
void Acquisition::ActiveSilicon1XCLD::stop_stream_read() {
    PHX_StreamRead(this->_handle, PHX_STOP, NULL);
}
void Acquisition::ActiveSilicon1XCLD::abort_stream_read() {
    PHX_StreamRead(this->_handle, PHX_ABORT, NULL);
}
