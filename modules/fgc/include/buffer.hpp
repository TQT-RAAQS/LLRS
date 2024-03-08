#pragma once

#include <wchar.h>
#include <DVPAPI.h>
#include <assert.h>

#include <vector>
#ifdef _WIN32
   #include <dvpapi_d3d11.h>
   #include <dvpapi_d3d9.h>
   #include <GL/glew.h>
#endif
#include <dvpapi_cuda.h>

#include <GL/glu.h>
#include <dvpapi_gl.h>

typedef enum {
   GPUAPI_UNKNOWN = 0,
   GPUAPI_GL,
   GPUAPI_DX9,
   GPUAPI_DX11,
   GPUAPI_CUDA
} etGPUAPI;

#ifdef WIN32
#include <windows.h>
#endif

typedef enum {
   BUFFER_STAT_OK                      = 0,
   BUFFER_STAT_INVALID_PARAMETER       = DVP_STATUS_INVALID_PARAMETER,
   BUFFER_STAT_UNSUPPORTED             = DVP_STATUS_UNSUPPORTED,
   BUFFER_STAT_END_ENUMERATION         = DVP_STATUS_END_ENUMERATION,
   BUFFER_STAT_INVALID_DEVICE          = DVP_STATUS_INVALID_DEVICE,
   BUFFER_STAT_OUT_OF_MEMORY           = DVP_STATUS_OUT_OF_MEMORY,
   BUFFER_STAT_INVALID_OPERATION       = DVP_STATUS_INVALID_OPERATION,
   BUFFER_STAT_TIMEOUT                 = DVP_STATUS_TIMEOUT,
   BUFFER_STAT_INVALID_CONTEXT         = DVP_STATUS_INVALID_CONTEXT,
   BUFFER_STAT_INVALID_RESOURCE_TYPE   = DVP_STATUS_INVALID_RESOURCE_TYPE,
   BUFFER_STAT_FORMAT_OR_TYPE          = DVP_STATUS_INVALID_FORMAT_OR_TYPE,
   BUFFER_STAT_DEVICE_UNINITIALIZED    = DVP_STATUS_DEVICE_UNINITIALIZED,
   BUFFER_STAT_UNSIGNALED              = DVP_STATUS_UNSIGNALED,
   BUFFER_STAT_SYNC_ERROR              = DVP_STATUS_SYNC_ERROR,
   BUFFER_STAT_SYNC_STILL_BOUND        = DVP_STATUS_SYNC_STILL_BOUND,
   BUFFER_STAT_INVALID_ENUM,
   BUFFER_STAT_RUNTIME_ERROR,
   BUFFER_STAT_UNKNOWN_ERROR           = -1
} etBufferStat;

typedef struct gpuObjectDescRec {
   bool     fUseTexture;      /* Using buffer or not */
   /* If Texture */
   uint32_t dwWidth;          /* Buffer Width */
   uint32_t dwHeight;         /* Buffer Height */
   uint32_t dwNumChannels;    /* To be populated by CUDA */
   uint32_t dwFormat;         /* To be populated by GL/D3D/CUDA */
   uint32_t dwType;           /* To be populated by GL */
   /* If buffer */
   uint32_t dwSize;           /* Specifies the surface size if it's non renderable format */
} gpuObjectDesc;

typedef struct syncInfoRec {
   volatile uint32_t   *pdwSem; 
   volatile uint32_t   *pdwSemOrg;
   volatile uint32_t    dwReleaseValue; 
   volatile uint32_t    dwAcquireValue;
   DVPSyncObjectHandle  dvpSyncObj;
} syncInfo;

class CBuffer
{
public:
   /**
    * Constructor
    * @param [in] eGPUAPI Selects the GPU API to be used
    * @param [in] device Optional pointer to Direct3D device, should be NULL when other GPU API used
    * @return 
   */
   CBuffer(etGPUAPI eGPUAPI, void *pvDevice);

   /**
    * Destructor
   */
   virtual ~CBuffer();

   /**
    * @brief Initialise this object with the GPU object provided (allocates system memory for DMA, semaphore, etc.).
    * @param desc [in]        Description of the GPU object
    * @param pGPUObject [in]  Pointer to the GPU object
    * @return etBufferStat    Error status
   */
   etBufferStat Init(gpuObjectDesc &desc, void *pvGPUObject, int nBitDepth = 8);

   /**
    * Undo what Init() did. 
    * @return etBufferStat    Error status
   */
   etBufferStat DeInit();

   /**
    * Close DVP device after DeInit() has been called on all buffers.
    * @return etBufferStat    Error status
   */
   etBufferStat CloseDevice();

   /**
    * Setup a DMA to transfer data to GPU and blocks until transfer completes. The transfer starts when @TransferChunk() is called. 
    * Should be called once for each PHX_INTRPT_BUFFER_READY event.
    * @return etBufferStat    Error status
   */
   etBufferStat GetFrame();

   /**
    * Trigger the DMA transfer that was setup by @GetFrame(). Should be called for each chunk of the buffer to transfer; chunks are 
    * not currently supported and it should be called for each PHX_INTRPT_BUFFER_READY event. Call before \GetFrame() if both are
    * called from the same thread.
    * @return etBufferStat    Error status
   */
   etBufferStat TransferChunk();

   /**
    * Inform DVP library that GPU is no longer used by application.
    * @return etBufferStat    Error status
   */
   etBufferStat EndFrame();
   
   /**
    * Retrieve address of the buffer in system memory associated with the GPU object.
    * @return void * Address of buffer
   */
   void *GetSystemMemory();

   /**
    * Return pointer to GPU object
    * @return void * Pointer to the GPU object; application should cast this to appropriate GPU object type.
   */
   void *GetGPUObject();

   /**
    * Return stride of system memory buffer required by DVP library; this should be used to set parameter
    * PHX_BUF_DST_XLENGTH in the PHX library.
    * @return std::uint32_t The stride of the buffer
   */
   uint32_t GetBufferStride() const {return m_dwStride;} /* Returns the stride of the buffer in system memory */

   /**
    * Return the size in bytes of the buffer allocated in system memory.
    * @return std::uint32_t Size of buffer
   */
   uint32_t GetBufferSize() const   {return (uint32_t)m_nSize;} /* Returns the size of the buffer in system memory */

protected:
   etGPUAPI          m_eGPUAPIUsed;
   void             *m_pvD3DDevice;

   bool              m_fTexture;          /* 1: texture, unformatted buffer otherwise */

   void             *m_pvSysMemAlloc;     /* Non-aligned memory */
   void             *m_pvSysMemBuffer;    /* Aligned memory (same as sysMemAlloc for windows) */

   DVPBufferHandle   m_hDvpSysMemHandle;
   syncInfo          m_ExtSync;

   union utGPUTextureHandleType {
     #ifdef _WIN32
      ID3D11Texture2D        *gpuTextureHandleD3D11;
      IDirect3DTexture9      *gpuTextureHandleD3D9;
     #endif
      CUarray                 gpuTextureHandleCUDA;
      GLuint                  gpuTextureHandleGL;
   };

   union utGPUBufferHandleType {
    #ifdef _WIN32
      ID3D11Buffer           *gpuBufferHandleD3D11;
      IDirect3DVertexBuffer9 *gpuBufferHandleD3D9;
    #endif
      CUdeviceptr             gpuBufferHandleCUDA;
      GLuint                  gpuBufferHandleGL;
   };

   union utGPUTextureHandleType  m_hGpuTextureHandle;
   union utGPUBufferHandleType   m_hGpuBufferHandle;

   DVPBufferHandle   m_hDvpGpuObjectHandle;
   syncInfo          m_GpuSync;

   uint32_t          m_dwWidth;           /* Image width */
   uint32_t          m_dwHeight;          /* Image height */
   uint32_t          m_dwStride;          /* Stride of buffer required by GPU */
   uint32_t          m_dwChunks;
   size_t            m_nSize;             /* Size in bytes, when using buffer instead of texture */

   static void      *m_asGPUMemInfo[5];   /* Array of the info for each tech */

   /* Blocks until DVP's DMA to GPU object is complete. */
   etBufferStat WaitGPUDoneWithObject();

   /* Initialises the stGPUMemInfo for the tech if not already done and returns it. */
   static etBufferStat GetGPUMemInfo(etGPUAPI eGPUAPI, void *pD3DDevice, struct stGPUMemInfo **ppsGPUMemInfo);
};
