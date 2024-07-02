#ifndef ACTIVESILICON_1XCLD_HPP_
#define ACTIVESILICON_1XCLD_HPP_

#define _PHX_LINUX
#include "phx_api.h"
#include "settings.hpp"
#include <boost/asio.hpp>
#include <iostream>
#include <png.h>
#include <thread>

#ifdef ENABLE_CUDA
#include "buffer.hpp"
#include <cuda.h>
#endif

class ActiveSilicon1XCLD {

  public:
    ActiveSilicon1XCLD() {}
    ~ActiveSilicon1XCLD();
    ActiveSilicon1XCLD(const ActiveSilicon1XCLD &) = delete;
    ActiveSilicon1XCLD &operator=(const ActiveSilicon1XCLD &) = delete;

    void setup(uint32_t roi_width, uint32_t roi_height, int acquisition_timeout,
               uint32_t roi_xoffset = 0, uint32_t roi_yoffset = 0,
               uint32_t vbin = 1, uint32_t hbin = 1);
    std::vector<uint16_t> acquire_single_image();
    std::vector<uint16_t> acquire_stored_image(const char *filename);

    /*
     * Closes the board handle
     * Returns: True if successful, false otherwise
     */
    bool destroy_handle();

    /*
     * Begins the acquisition process. After the call, control is returned to
     * this application. Use PHX_check_and_wait to check if an image is ready to
     * be acquired. This function should be called before calling
     * get_current_buffer, and the program should always end with a call to
     * stop_stream_read.
     */
    void start_stream_read();

    /*
     * Stops the acquisition process after the current image is acquired.
     * This function should always be called after the necessary images are
     * acquired.
     */
    void stop_stream_read();

    /*
     * Aborts the acquisition process immediately
     */
    void abort_stream_read();

    /*
     * Waits indefinitely for an event to occur. Only return control to the
     * user's application when an event occurs or times out.
     */
    int32_t check_and_wait();

    /*
     * Gets the next unprocessed image buffer. This function will return the
     * same image until release_current_buffer is called.
     * Returns: The image stored in the buffer as a vector of shorts.
     */
    std::vector<uint16_t> get_current_buffer();

    /*
     * Releases the current image buffer to be reused by the hardware.
     */
    void release_current_buffer();

  private:
    tHandle _handle;
    uint32_t _roi_width;
    uint32_t _roi_height;
    uint32_t _roi_xoffset;
    uint32_t _roi_yoffset;
    uint32_t _vbin;
    uint32_t _hbin;
    int _acquisition_timeout;
    stImageBuff *_image_buffers = new stImageBuff[2];

#ifdef ENABLE_CUDA
    // For GPU
    CBuffer gpu_buffer = CBuffer(GPUAPI_CUDA, NULL);
    short *gpu_buffers_src;
#endif

    void init_image_buffer();
};

#endif
