// ----- This file contains functions that are used by multiple CUDA RDMA
// examples -----

#include "../c_header/dlltyp.h"
#include <cufft.h>

// ----- Init CUDA device without RDMA support (e.g. on Windows) ----
bool bInitCUDADevice(int lCUDADevIdx);

// ----- This function creates a DMA buffer on the GPU and sets it up for use as
// RDMA buffer -----
// -- lCudaDevIdx:              in      Index of CUDA device in the system,
// starting with 0
// -- qwDMABufferSize_bytes:    in      Size of DMA buffer that should be
// created on GPU
// -- returns address of created DMA buffer on GPU
void *pvGetRDMABuffer(int lCUDADevIdx, size_t qwDMABufferSize_bytes);

// ---returns description for CUDA FFT error
const char *szCudaGetErrorText(cufftResult eError);
