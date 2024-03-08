#include "buffer.hpp"

#include <wchar.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include "trace.hpp"

using namespace std;

#define MEM_WR32(a, d) do {*(volatile unsigned int *)(a) = (d);} while (0)

#define RETURN_ON_ERROR(function) {                \
   etBufferStat eStat = (etBufferStat)(function);  \
   if (BUFFER_STAT_OK != eStat) {                  \
      return eStat;                                \
   }                                               \
}

/* Stores info valid for the current GPU context (GL or else) */
typedef struct stGPUMemInfo {
   bool     fInitialised;
   uint32_t dwBufferAddrAlignment;
   uint32_t dwBufferGPUStrideAlignment;
   uint32_t dwSemaphoreAddrAlignment;
   uint32_t dwSemaphoreAllocSize;
   uint32_t dwSemaphorePayloadOffset;
   uint32_t dwSemaphorePayloadSize;
} stGPUMemInfo;

/*
CUDAArrayParamsToDVPParams()
 * Converts CUDA format to DVP format and type.
 */
etBufferStat CUDAArrayParamsToDVPParams(
   CUarray_format    cudaInFormat,
   int               nInNumChannels,
   DVPBufferFormats *pdvpOutFormat,
   DVPBufferTypes   *pdvpOutType
)
{
   if (!pdvpOutFormat || !pdvpOutType) {
      return BUFFER_STAT_INVALID_PARAMETER;
   }

   switch(cudaInFormat) {
      case CU_AD_FORMAT_UNSIGNED_INT8:
         *pdvpOutType = DVP_UNSIGNED_BYTE;
         break;

      case CU_AD_FORMAT_SIGNED_INT8:
         *pdvpOutType = DVP_BYTE;
         break;

      case CU_AD_FORMAT_UNSIGNED_INT16:
         *pdvpOutType = DVP_UNSIGNED_SHORT;
         break;

      case CU_AD_FORMAT_SIGNED_INT16:
         *pdvpOutType = DVP_SHORT;
         break;

      case CU_AD_FORMAT_HALF:
         *pdvpOutType = DVP_HALF_FLOAT;
         break;

      case CU_AD_FORMAT_UNSIGNED_INT32:
         *pdvpOutType = DVP_UNSIGNED_INT;
         break;

      case CU_AD_FORMAT_SIGNED_INT32:
         *pdvpOutType = DVP_INT;
         break;

      case CU_AD_FORMAT_FLOAT:
         *pdvpOutType = DVP_FLOAT;
         break;

      default:
         return BUFFER_STAT_INVALID_ENUM;
   }

   switch(nInNumChannels) {
      case 1:
         *pdvpOutFormat = DVP_CUDA_1_CHANNEL;
         break;

      case 2:
         *pdvpOutFormat = DVP_CUDA_2_CHANNELS;
         break;

      case 4:
         *pdvpOutFormat = DVP_CUDA_4_CHANNELS;
         break;

      default:
         return BUFFER_STAT_INVALID_ENUM;
   }

   return BUFFER_STAT_OK;
}

/*
OGLTextureParamsToDVPParams()
 * Converts OpenGL type and format to DVP type and format.
 */
etBufferStat OGLTextureParamsToDVPParams(
   GLenum            glInFormat,
   GLenum            glInType,
   DVPBufferFormats *pdvpOutFormat,
   DVPBufferTypes   *pdvpOutType
)
{
   if (!pdvpOutFormat || !pdvpOutType) {
      return BUFFER_STAT_INVALID_PARAMETER;
   }

   switch(glInType) {
      case GL_UNSIGNED_BYTE:
         *pdvpOutType = DVP_UNSIGNED_BYTE;
         break;

      case GL_BYTE:
         *pdvpOutType = DVP_BYTE;
         break;

      case GL_UNSIGNED_BYTE_3_3_2:
         *pdvpOutType = DVP_UNSIGNED_BYTE_3_3_2;
         break;

      case GL_UNSIGNED_BYTE_2_3_3_REV:
         *pdvpOutType = DVP_UNSIGNED_BYTE_3_3_2;
         break;

      case GL_UNSIGNED_SHORT:
         *pdvpOutType = DVP_UNSIGNED_SHORT;
         break;

      case GL_SHORT:
         *pdvpOutType = DVP_SHORT;
         break;

      case GL_UNSIGNED_SHORT_5_6_5:
         *pdvpOutType = DVP_UNSIGNED_SHORT_5_6_5;
         break;

      case GL_UNSIGNED_SHORT_5_6_5_REV:
         *pdvpOutType = DVP_UNSIGNED_SHORT_5_6_5_REV;
         break;

      case GL_UNSIGNED_SHORT_4_4_4_4:
         *pdvpOutType = DVP_UNSIGNED_SHORT_4_4_4_4;
         break;

      case GL_UNSIGNED_SHORT_4_4_4_4_REV:
         *pdvpOutType = DVP_UNSIGNED_SHORT_4_4_4_4_REV;
         break;

      case GL_UNSIGNED_SHORT_5_5_5_1:
         *pdvpOutType = DVP_UNSIGNED_SHORT_5_5_5_1;
         break;

      case GL_UNSIGNED_SHORT_1_5_5_5_REV:
         *pdvpOutType = DVP_UNSIGNED_SHORT_1_5_5_5_REV;
         break;

      case GL_HALF_FLOAT:
         *pdvpOutType = DVP_HALF_FLOAT;
         break;

      case GL_UNSIGNED_INT:
         *pdvpOutType = DVP_UNSIGNED_INT;
         break;

      case GL_INT:
         *pdvpOutType = DVP_INT;
         break;

      case GL_FLOAT:
         *pdvpOutType = DVP_FLOAT;
         break;

      case GL_UNSIGNED_INT_8_8_8_8:
         *pdvpOutType = DVP_UNSIGNED_INT_8_8_8_8;
         break;

      case GL_UNSIGNED_INT_8_8_8_8_REV:
         *pdvpOutType = DVP_UNSIGNED_INT_8_8_8_8_REV;
         break;

      case GL_UNSIGNED_INT_10_10_10_2:
         *pdvpOutType = DVP_UNSIGNED_INT_10_10_10_2;
         break;

      case GL_UNSIGNED_INT_2_10_10_10_REV:
         *pdvpOutType = DVP_UNSIGNED_INT_2_10_10_10_REV;
         break;

      default:
         return BUFFER_STAT_INVALID_ENUM;
   }

   switch(glInFormat) {
      case GL_DEPTH_COMPONENT:
         *pdvpOutFormat = DVP_DEPTH_COMPONENT;
         break;

      case GL_RGBA:
         *pdvpOutFormat = DVP_RGBA;
         break;

      case GL_BGRA:
         *pdvpOutFormat = DVP_BGRA;
         break;

      case GL_RED:
         *pdvpOutFormat = DVP_RED;
         break;

      case GL_GREEN:
         *pdvpOutFormat = DVP_GREEN;
         break;

      case GL_BLUE:
         *pdvpOutFormat = DVP_BLUE;
         break;

      case GL_ALPHA:
         *pdvpOutFormat = DVP_ALPHA;
         break;

      case GL_RGB:
         *pdvpOutFormat = DVP_RGB;
         break;

      case GL_BGR:
         *pdvpOutFormat = DVP_BGR;
         break;

      case GL_LUMINANCE:
         *pdvpOutFormat = DVP_LUMINANCE;
         break;

      case GL_LUMINANCE_ALPHA:
         *pdvpOutFormat = DVP_LUMINANCE_ALPHA;
         break;

      default:
         return BUFFER_STAT_INVALID_ENUM;
   }

   return BUFFER_STAT_OK;
}

/*
DVPBufferTypeToBytes()
 * Returns number of bytes for input argument type.
 * Returns 0 if unknown type.
 */
int DVPBufferTypeToBytes(DVPBufferTypes type)
{
   int nNumBytes;
   switch(type) {
      case DVP_BYTE:
      case DVP_UNSIGNED_BYTE:
      case DVP_UNSIGNED_BYTE_3_3_2:
      case DVP_UNSIGNED_BYTE_2_3_3_REV:
         nNumBytes = 1;
         break;

      case DVP_HALF_FLOAT:
      case DVP_SHORT:
      case DVP_UNSIGNED_SHORT:
      case DVP_UNSIGNED_SHORT_5_6_5:
      case DVP_UNSIGNED_SHORT_5_6_5_REV:
      case DVP_UNSIGNED_SHORT_4_4_4_4:
      case DVP_UNSIGNED_SHORT_4_4_4_4_REV:
      case DVP_UNSIGNED_SHORT_5_5_5_1:
      case DVP_UNSIGNED_SHORT_1_5_5_5_REV:
         nNumBytes = 2;
         break;

      case DVP_FLOAT:
      case DVP_INT:
      case DVP_UNSIGNED_INT:
      case DVP_UNSIGNED_INT_8_8_8_8:
      case DVP_UNSIGNED_INT_8_8_8_8_REV:
      case DVP_UNSIGNED_INT_10_10_10_2:
      case DVP_UNSIGNED_INT_2_10_10_10_REV:
         nNumBytes = 4;
         break;

      default:
         nNumBytes = 0;
         break;
   }

   return nNumBytes;
}

/*
DVPBufferFormatToNumChannels()
 * Returns number of channels for input argument format.
 * Returns 0 if unknown format.
 */
int DVPBufferFormatToNumChannels(DVPBufferFormats format)
{
   int numChannels;
   switch(format) {
      case DVP_DEPTH_COMPONENT:
      case DVP_RED:
      case DVP_GREEN:
      case DVP_BLUE:
      case DVP_ALPHA:
      case DVP_LUMINANCE:
      case DVP_CUDA_1_CHANNEL:
         numChannels = 1;
         break;

      case DVP_LUMINANCE_ALPHA:
      case DVP_CUDA_2_CHANNELS:
         numChannels = 2;
         break;

      case DVP_RGB:
      case DVP_BGR:
         numChannels = 3;
         break;

      case DVP_RGBA:
      case DVP_BGRA:
      case DVP_CUDA_4_CHANNELS:
         numChannels = 4;
         break;

      default:
         numChannels = 0;
         break;
   }
   return numChannels;
}

/*
SetupSyncObject()
 * Initialises a DVP sync object.
 */
etBufferStat SetupSyncObject(
   syncInfo            *psSyncInfo,
   const stGPUMemInfo  &sGPUMemInfo
)
{
   etBufferStat eReturnStatus = BUFFER_STAT_OK;

   DVPSyncObjectDesc syncObjectDesc;
   if (!sGPUMemInfo.dwSemaphoreAllocSize || !sGPUMemInfo.dwSemaphoreAddrAlignment || !psSyncInfo) {
      return BUFFER_STAT_INVALID_PARAMETER;
   }
   psSyncInfo->pdwSemOrg = (uint32_t *) malloc(sGPUMemInfo.dwSemaphoreAllocSize + sGPUMemInfo.dwSemaphoreAddrAlignment - 1);

   if (!psSyncInfo->pdwSemOrg) {
      return BUFFER_STAT_RUNTIME_ERROR;
   }

   /* Correct alignment */
   uint64_t qwVal = (uint64_t)psSyncInfo->pdwSemOrg;
   qwVal += sGPUMemInfo.dwSemaphoreAddrAlignment - 1;
   qwVal &= ~((uint64_t)sGPUMemInfo.dwSemaphoreAddrAlignment - 1);
   psSyncInfo->pdwSem = (uint32_t *) qwVal;

   /* Initialise members */
   MEM_WR32(psSyncInfo->pdwSem, 0);
   psSyncInfo->dwReleaseValue             = 0;
   psSyncInfo->dwAcquireValue             = 0;
   syncObjectDesc.sem                     = (uint32_t *)psSyncInfo->pdwSem;
   syncObjectDesc.externalClientWaitFunc  = NULL;
   syncObjectDesc.flags                   = 0;  /* DVP_SYNC_OBJECT_FLAGS_USE_EVENTS */

   eReturnStatus = (etBufferStat)dvpImportSyncObject(&syncObjectDesc, &psSyncInfo->dvpSyncObj);

   return eReturnStatus;
}

void *CBuffer::m_asGPUMemInfo[5] = {0, 0, 0, 0, 0};

CBuffer::CBuffer(
   etGPUAPI eGPUAPI,
   void *pvDevice
)
: m_eGPUAPIUsed(eGPUAPI)
, m_pvD3DDevice(pvDevice)
, m_fTexture(true)
, m_pvSysMemAlloc(0)
, m_pvSysMemBuffer(0)
, m_hDvpSysMemHandle(0)
, m_ExtSync()
, m_hDvpGpuObjectHandle(0)
, m_GpuSync()
, m_dwWidth(0)
, m_dwHeight(0)
, m_dwStride(0)
, m_dwChunks(0)
, m_nSize(0)
{
#ifdef _WIN32
   m_hGpuTextureHandle.gpuTextureHandleD3D11 = 0;
   m_hGpuBufferHandle.gpuBufferHandleD3D11   = 0;
#endif
}

CBuffer::~CBuffer()
{
}

etBufferStat CBuffer::Init(
   gpuObjectDesc &desc,
   void          *pvGPUObject,
   int            nBitDepth   /* = 8 */
)
{
   DVPSysmemBufferDesc  sysMemBuffersDesc; /* struct with info about buffer (stride etc.) */
   stGPUMemInfo        *psGPUMemInfo   = 0;

   RETURN_ON_ERROR(GetGPUMemInfo(m_eGPUAPIUsed, m_pvD3DDevice, &psGPUMemInfo));
   const stGPUMemInfo &sGPUMemInfo = *psGPUMemInfo;

   if (0 == &sGPUMemInfo) {
      return BUFFER_STAT_UNKNOWN_ERROR;
   }

   if (!desc.fUseTexture) {
      /* Create sysmem and sync object etc. */
      sysMemBuffersDesc.format   = DVP_BUFFER;
      sysMemBuffersDesc.type     = DVP_UNSIGNED_BYTE;
      sysMemBuffersDesc.size     = desc.dwSize;

      m_dwWidth   = desc.dwWidth;
      m_dwHeight  = desc.dwHeight;
      m_dwStride  = m_dwWidth;
      m_dwChunks  = 1;
      m_nSize     = sysMemBuffersDesc.size;

   } else {
      switch (m_eGPUAPIUsed) {
         case GPUAPI_DX11:
            sysMemBuffersDesc.format   = DVP_RGBA;
            sysMemBuffersDesc.type     = DVP_UNSIGNED_BYTE;
            break;

         case GPUAPI_DX9:
            sysMemBuffersDesc.format   = DVP_LUMINANCE;
            if (16 == nBitDepth) {
               sysMemBuffersDesc.type  = DVP_UNSIGNED_SHORT;
            } else {
               sysMemBuffersDesc.type  = DVP_UNSIGNED_BYTE;
            }
            break;

         case GPUAPI_CUDA:
            RETURN_ON_ERROR(CUDAArrayParamsToDVPParams((CUarray_format)desc.dwFormat, desc.dwNumChannels, &sysMemBuffersDesc.format, &sysMemBuffersDesc.type));
            break;

         default:
            if (16 == nBitDepth) {
               desc.dwType = GL_UNSIGNED_SHORT;
            } else {
               desc.dwType = GL_UNSIGNED_BYTE;
            }
            RETURN_ON_ERROR(OGLTextureParamsToDVPParams((GLenum)desc.dwFormat,(GLenum)desc.dwType, &sysMemBuffersDesc.format, &sysMemBuffersDesc.type));
            break;
      }

      int nNumBytesPerChannel    = DVPBufferTypeToBytes(sysMemBuffersDesc.type);
      int nNumChannels           = DVPBufferFormatToNumChannels(sysMemBuffersDesc.format);

      if (!nNumBytesPerChannel || !nNumChannels) {
         return BUFFER_STAT_INVALID_PARAMETER;
      }

      sysMemBuffersDesc.width    = desc.dwWidth;
      sysMemBuffersDesc.height   = desc.dwHeight;

      uint32_t dwBufferStride    = desc.dwWidth * nNumChannels * nNumBytesPerChannel;
      dwBufferStride             += sGPUMemInfo.dwBufferGPUStrideAlignment - 1;
      dwBufferStride             &= ~(sGPUMemInfo.dwBufferGPUStrideAlignment - 1);

      sysMemBuffersDesc.stride   = dwBufferStride;
      sysMemBuffersDesc.size     = dwBufferStride * desc.dwHeight;

      m_dwWidth   = desc.dwWidth;
      m_dwHeight  = desc.dwHeight;
      m_dwStride  = dwBufferStride;
      m_dwChunks  = 1;
      m_nSize     = sysMemBuffersDesc.size;
   }

   m_fTexture = desc.fUseTexture;

   switch(m_eGPUAPIUsed) {
      #ifdef _WIN32
      case GPUAPI_DX11:
         if(desc.fUseTexture) {
            m_hGpuTextureHandle.gpuTextureHandleD3D11 = *((ID3D11Texture2D  **)pvGPUObject);
         } else {
            m_hGpuBufferHandle.gpuBufferHandleD3D11   = *((ID3D11Buffer **)pvGPUObject);
         } break;

      case GPUAPI_DX9:
         if(desc.fUseTexture) {
            m_hGpuTextureHandle.gpuTextureHandleD3D9  = *((IDirect3DTexture9  **)pvGPUObject);
         } else {
            m_hGpuBufferHandle.gpuBufferHandleD3D9    = *((IDirect3DVertexBuffer9 **)pvGPUObject);
         } break;
      #endif
      case GPUAPI_CUDA:
         if(desc.fUseTexture) {
            m_hGpuTextureHandle.gpuTextureHandleCUDA  = *((CUarray *)pvGPUObject);
         } else {
            m_hGpuBufferHandle.gpuBufferHandleCUDA    = *((CUdeviceptr *)pvGPUObject);
         } break;

      default:
         if(desc.fUseTexture) {
            m_hGpuTextureHandle.gpuTextureHandleGL    = *((GLuint *)pvGPUObject);
         } else {
            m_hGpuBufferHandle.gpuBufferHandleGL      = *((GLuint *)pvGPUObject);
         } break;
   }
#ifdef _WIN32
   m_pvSysMemBuffer = VirtualAlloc(NULL, sysMemBuffersDesc.size, MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE);
#else 
   m_pvSysMemBuffer = malloc(sysMemBuffersDesc.size);
#endif
   if (!m_pvSysMemBuffer) {
      return BUFFER_STAT_RUNTIME_ERROR;
   }

   /* Create a DVP buffer pointing to the allocated buffer */
   sysMemBuffersDesc.bufAddr = m_pvSysMemBuffer;
   RETURN_ON_ERROR(dvpCreateBuffer(&sysMemBuffersDesc, &m_hDvpSysMemHandle));
  
   switch(m_eGPUAPIUsed) { 
#ifdef _WIN32
      case GPUAPI_DX11:
         RETURN_ON_ERROR(dvpBindToD3D11Device(m_hDvpSysMemHandle, (ID3D11Device *)m_pvD3DDevice));
         break;

      case GPUAPI_DX9:
         RETURN_ON_ERROR(dvpBindToD3D9Device(m_hDvpSysMemHandle, (IDirect3DDevice9 *)m_pvD3DDevice));
         break;
 #endif  
      case GPUAPI_CUDA:
         RETURN_ON_ERROR(dvpBindToCUDACtx(m_hDvpSysMemHandle));
         break;

      default:
         RETURN_ON_ERROR(dvpBindToGLCtx(m_hDvpSysMemHandle));
         break;
   }

   /* Create necessary sysmem sync objects */
   RETURN_ON_ERROR(SetupSyncObject(&m_ExtSync, sGPUMemInfo));

   switch (m_eGPUAPIUsed) {
   #ifdef _WIN32    	
      case GPUAPI_DX11:
         if (desc.fUseTexture) {
            RETURN_ON_ERROR(dvpCreateGPUD3D11Resource((ID3D11Resource *)m_hGpuTextureHandle.gpuTextureHandleD3D11, &m_hDvpGpuObjectHandle));
         } else {
            RETURN_ON_ERROR(dvpCreateGPUD3D11Resource((ID3D11Resource *)m_hGpuBufferHandle.gpuBufferHandleD3D11, &m_hDvpGpuObjectHandle));
         } break;

      case GPUAPI_DX9:
         if (desc.fUseTexture) {
            RETURN_ON_ERROR(dvpCreateGPUD3D9Resource((IDirect3DResource9 *)m_hGpuTextureHandle.gpuTextureHandleD3D9, &m_hDvpGpuObjectHandle));
         } else {
            RETURN_ON_ERROR(dvpCreateGPUD3D9Resource((IDirect3DResource9 *)m_hGpuBufferHandle.gpuBufferHandleD3D9, &m_hDvpGpuObjectHandle));
         } break;
#endif
      case GPUAPI_CUDA:
         if (desc.fUseTexture) {
            RETURN_ON_ERROR(dvpCreateGPUCUDAArray(m_hGpuTextureHandle.gpuTextureHandleCUDA, &m_hDvpGpuObjectHandle));
         } else {
            RETURN_ON_ERROR(dvpCreateGPUCUDADevicePtr(m_hGpuBufferHandle.gpuBufferHandleCUDA, &m_hDvpGpuObjectHandle));
         } break;

      default:
         if (desc.fUseTexture) {
            RETURN_ON_ERROR(dvpCreateGPUTextureGL(m_hGpuTextureHandle.gpuTextureHandleGL, &m_hDvpGpuObjectHandle));
         } else {
            RETURN_ON_ERROR(dvpCreateGPUBufferGL(m_hGpuBufferHandle.gpuBufferHandleGL, &m_hDvpGpuObjectHandle));
         } break;
   }

   /* Create necessary gpu mem sync objects */
   RETURN_ON_ERROR(SetupSyncObject(&m_GpuSync, sGPUMemInfo));

   return BUFFER_STAT_OK;
}

etBufferStat CBuffer::TransferChunk()
{
   /* Update the release value */
   m_ExtSync.dwReleaseValue++;

   /* Update the semaphore when each chunk is DMAed */
   MEM_WR32(m_ExtSync.pdwSem, m_ExtSync.dwReleaseValue);

   return BUFFER_STAT_OK;
}

#define LINED_COPY
etBufferStat CBuffer::GetFrame()
{
   SETUP_TRACE

   RETURN_ON_ERROR(dvpBegin());

   /* We want to acquire the previous transfer's release value on the first call
    * to make sure that we don't corrupt the previous data. This is likely to 
    * happen if the input is faster than the app processing.
    * Make sure we are done with the pending GPU DMAs.
    */

   /* Wait for GPU API to be done rendering buffer (signalled by MapBufferEndAPI() before we attempt to DMA data into it */
   START_TRACE
      RETURN_ON_ERROR(dvpMapBufferWaitDVP(m_hDvpGpuObjectHandle));
   END_TRACE(dvpMapBufferWaitDVP);

   if (m_fTexture) {
#ifdef LINED_COPY
      uint32_t dwCopiedLines     = 0;
      uint32_t dwNumLinesPerCopy = m_dwHeight / m_dwChunks;
      while (dwCopiedLines < m_dwHeight) {

         uint32_t dwLinesToCopy = (m_dwHeight - dwCopiedLines > dwNumLinesPerCopy ? dwNumLinesPerCopy : m_dwHeight - dwCopiedLines);

         m_ExtSync.dwAcquireValue++;
         m_GpuSync.dwReleaseValue++;

         START_TRACE
            RETURN_ON_ERROR(dvpMemcpyLined(m_hDvpSysMemHandle, m_ExtSync.dvpSyncObj, m_ExtSync.dwAcquireValue, DVP_TIMEOUT_IGNORED,
                                          m_hDvpGpuObjectHandle, m_GpuSync.dvpSyncObj, m_GpuSync.dwReleaseValue, dwCopiedLines, dwLinesToCopy));
         END_TRACE(dvpMemcpyLined);

         dwCopiedLines += dwLinesToCopy;
      }
#else
      uint32_t dwCopiedLines = 0;
      uint32_t dwNumLinesPerCopy = m_dwHeight / m_dwChunks;

      while (dwCopiedLines < m_dwHeight) {
         uint32_t dwLinesToCopy = (m_dwHeight - dwCopiedLines > dwNumLinesPerCopy ? dwNumLinesPerCopy : m_dwHeight - dwCopiedLines);
         m_ExtSync.dwAcquireValue++;
         m_GpuSync.dwReleaseValue++;

         START_TRACE
            RETURN_ON_ERROR(dvpMemcpy2D(m_hDvpSysMemHandle, m_ExtSync.dvpSyncObj, m_ExtSync.dwAcquireValue, DVP_TIMEOUT_IGNORED,
                                       m_hDvpGpuObjectHandle, m_GpuSync.dvpSyncObj, m_GpuSync.dwReleaseValue, 0, dwCopiedLines, dwLinesToCopy, m_dwWidth));
         END_TRACE(dvpMemcpy2D);

         dwCopiedLines += dwLinesToCopy;
      }
#endif
   } else {
      uint32_t dwCopiedSize   = 0;
      uint32_t dwChunkSize    = (uint32_t)m_nSize / m_dwChunks;

      while (dwCopiedSize < (uint32_t)m_nSize) {
         uint32_t dwCopiedChunkSize = ((uint32_t)m_nSize - dwCopiedSize > dwChunkSize ? dwChunkSize : (uint32_t)m_nSize - dwCopiedSize);
         m_ExtSync.dwAcquireValue++;
         m_GpuSync.dwReleaseValue++;

         START_TRACE
            RETURN_ON_ERROR(dvpMemcpy(m_hDvpSysMemHandle, m_ExtSync.dvpSyncObj, m_ExtSync.dwAcquireValue, DVP_TIMEOUT_IGNORED,
                                    m_hDvpGpuObjectHandle, m_GpuSync.dvpSyncObj, m_GpuSync.dwReleaseValue, dwCopiedSize, dwCopiedSize, dwCopiedChunkSize));
         END_TRACE(dvpMemcpy);

         dwCopiedSize += dwCopiedChunkSize;
      }
   }

   START_TRACE
      /* Signal to GPU API that the GPU Object can now be used (waited for via MapBufferWaitAPI()) */
      RETURN_ON_ERROR(dvpMapBufferEndDVP(m_hDvpGpuObjectHandle));
   END_TRACE(dvpMapBufferEndDVP);
   
   RETURN_ON_ERROR(dvpEnd());

   RETURN_ON_ERROR(WaitGPUDoneWithObject());

   return BUFFER_STAT_OK;
}

etBufferStat CBuffer::WaitGPUDoneWithObject()
{
   SETUP_TRACE
   /* Wait for DMA to GPU to have completed and GPU object to be usable (signaled via dvpMapBufferEndDVP()) */
   if (m_eGPUAPIUsed == GPUAPI_CUDA) {
      CUstream stream = 0;
      START_TRACE
         RETURN_ON_ERROR(dvpMapBufferWaitCUDAStream(m_hDvpGpuObjectHandle, stream));
      END_TRACE(dvpMapBufferWaitCUDAStream);
   } else {
      START_TRACE
         RETURN_ON_ERROR(dvpMapBufferWaitAPI(m_hDvpGpuObjectHandle));
      END_TRACE(dvpMapBufferWaitAPI);
   }

   return BUFFER_STAT_OK;
}

etBufferStat CBuffer::EndFrame()
{
   SETUP_TRACE
   if (m_eGPUAPIUsed == GPUAPI_CUDA) {
      CUstream stream = 0;
      START_TRACE      
         RETURN_ON_ERROR(dvpMapBufferEndCUDAStream(m_hDvpGpuObjectHandle, stream));
      END_TRACE(dvpMapBufferEndCUDAStream);
   } else {
      START_TRACE
         RETURN_ON_ERROR(dvpMapBufferEndAPI(m_hDvpGpuObjectHandle));
      END_TRACE(dvpMapBufferEndAPI);
   }

   return BUFFER_STAT_OK;
}

etBufferStat CBuffer::GetGPUMemInfo(
   etGPUAPI       eGPUAPI,
   void          *pD3DDevice,
   stGPUMemInfo **ppsGPUMemInfo
)
{
   stGPUMemInfo *psGPUMemInfo = (stGPUMemInfo *)m_asGPUMemInfo[eGPUAPI];

   if (!psGPUMemInfo) {
      psGPUMemInfo                        = new stGPUMemInfo;
      m_asGPUMemInfo[eGPUAPI]             = psGPUMemInfo;
      stGPUMemInfo &sGPUMemInfo           = *psGPUMemInfo;
      sGPUMemInfo.dwBufferAddrAlignment   = 4096;

      switch (eGPUAPI) {
 #ifdef _WIN32  
         case GPUAPI_DX11:
         {
            ID3D11Device *pD3Ddev = (ID3D11Device *)pD3DDevice;
            RETURN_ON_ERROR(dvpInitD3D11Device(pD3Ddev,0))
            RETURN_ON_ERROR(dvpGetRequiredConstantsD3D11Device(&sGPUMemInfo.dwBufferAddrAlignment, &sGPUMemInfo.dwBufferGPUStrideAlignment, &sGPUMemInfo.dwSemaphoreAddrAlignment,
                                                               &sGPUMemInfo.dwSemaphoreAllocSize, &sGPUMemInfo.dwSemaphorePayloadOffset, &sGPUMemInfo.dwSemaphorePayloadSize, pD3Ddev));
         } break;

         case GPUAPI_DX9:
         {
            IDirect3DDevice9 *pD3Ddev = (IDirect3DDevice9 *)pD3DDevice;
            RETURN_ON_ERROR(dvpInitD3D9Device(pD3Ddev,0));
            RETURN_ON_ERROR(dvpGetRequiredConstantsD3D9Device(&sGPUMemInfo.dwBufferAddrAlignment, &sGPUMemInfo.dwBufferGPUStrideAlignment, &sGPUMemInfo.dwSemaphoreAddrAlignment,
                                                               &sGPUMemInfo.dwSemaphoreAllocSize, &sGPUMemInfo.dwSemaphorePayloadOffset, &sGPUMemInfo.dwSemaphorePayloadSize, pD3Ddev));
         } break;
#endif
         case GPUAPI_CUDA:
         {
            RETURN_ON_ERROR(dvpInitCUDAContext(DVP_DEVICE_FLAGS_SHARE_APP_CONTEXT));
            RETURN_ON_ERROR(dvpGetRequiredConstantsCUDACtx(&sGPUMemInfo.dwBufferAddrAlignment, &sGPUMemInfo.dwBufferGPUStrideAlignment, &sGPUMemInfo.dwSemaphoreAddrAlignment,
                                                            &sGPUMemInfo.dwSemaphoreAllocSize, &sGPUMemInfo.dwSemaphorePayloadOffset, &sGPUMemInfo.dwSemaphorePayloadSize));
         } break;

         case GPUAPI_GL:
         {
            RETURN_ON_ERROR(dvpInitGLContext(0));
            RETURN_ON_ERROR(dvpGetRequiredConstantsGLCtx(&sGPUMemInfo.dwBufferAddrAlignment, &sGPUMemInfo.dwBufferGPUStrideAlignment, &sGPUMemInfo.dwSemaphoreAddrAlignment,
                                                         &sGPUMemInfo.dwSemaphoreAllocSize, &sGPUMemInfo.dwSemaphorePayloadOffset, &sGPUMemInfo.dwSemaphorePayloadSize));
         } break;

         default:
            return BUFFER_STAT_UNSUPPORTED;
      }
   }

   *ppsGPUMemInfo = psGPUMemInfo;
   return BUFFER_STAT_OK;
}

etBufferStat CBuffer::DeInit()
{
   /* Unbind from source GPU */
 #ifdef _WIN32    
 if (m_eGPUAPIUsed == GPUAPI_DX11) {
      RETURN_ON_ERROR(dvpUnbindFromD3D11Device(m_hDvpSysMemHandle, (ID3D11Device *)m_pvD3DDevice));
   } else if (m_eGPUAPIUsed == GPUAPI_DX9) {
      RETURN_ON_ERROR(dvpUnbindFromD3D9Device(m_hDvpSysMemHandle, (IDirect3DDevice9 *)m_pvD3DDevice));
}
#endif
   if (m_eGPUAPIUsed == GPUAPI_CUDA) {
      RETURN_ON_ERROR(dvpUnbindFromCUDACtx(m_hDvpSysMemHandle));
   } else if(m_eGPUAPIUsed == GPUAPI_GL) {
      RETURN_ON_ERROR(dvpUnbindFromGLCtx(m_hDvpSysMemHandle));
   }

   RETURN_ON_ERROR(dvpDestroyBuffer(m_hDvpSysMemHandle));

   RETURN_ON_ERROR(dvpFreeSyncObject(m_ExtSync.dvpSyncObj));

   RETURN_ON_ERROR(dvpFreeBuffer(m_hDvpGpuObjectHandle));

   RETURN_ON_ERROR(dvpFreeSyncObject(m_GpuSync.dvpSyncObj));

#if defined(WIN32)
   if (m_pvSysMemBuffer) {
      VirtualFree(m_pvSysMemBuffer, 0, MEM_RELEASE);
      m_pvSysMemBuffer = NULL;
   }
#else
   free(m_pvSysMemAlloc);
   m_pvSysMemAlloc   = 0;
   m_pvSysMemBuffer  = 0;
#endif

   return BUFFER_STAT_OK;
}

etBufferStat CBuffer::CloseDevice()
{
   if (m_eGPUAPIUsed == GPUAPI_CUDA) {
      dvpCloseCUDAContext();
   }

#ifdef _WIN32  
   if (m_eGPUAPIUsed == GPUAPI_DX11) {
      dvpCloseD3D11Device((ID3D11Device *)m_pvD3DDevice);
   } else if (m_eGPUAPIUsed == GPUAPI_DX9) {
      dvpCloseD3D9Device((IDirect3DDevice9 *)m_pvD3DDevice);
   }
#endif

   if(m_eGPUAPIUsed == GPUAPI_GL){
      dvpCloseGLContext();
   }

   return BUFFER_STAT_OK;
}

void *CBuffer::GetGPUObject()
{
   if (m_fTexture) {
      return &m_hGpuTextureHandle;
   } else {
      return &m_hGpuBufferHandle;
   }
}

void *CBuffer::GetSystemMemory()
{
   return m_pvSysMemBuffer;
}
