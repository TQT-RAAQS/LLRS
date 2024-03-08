#ifndef TLACTIVESILICON_H_
#define TLACTIVESILICON_H_ 1

#include "GenTL.h"

#ifdef __cplusplus
extern "C" {
   namespace ActiveSilicon {
      namespace GenTL {
#endif

         /* ID values for the GenICam_GenTL::Client::EVENT_ERROR event.
          * The event ID and its description (message string) can be retrieved using the EventGetDataInfo() function.
          */
         enum AS_EVENT_ERROR_ID_LIST
         {
            AS_EVENT_ERROR_ID_ERROR                      = -10001,   /* Unspecified runtime error */
            AS_EVENT_ERROR_ID_PHX_INTRPT_FIFO_OVERFLOW   = -10002,   /* PHX FIFO overflow event detected in callback */
            AS_EVENT_ERROR_ID_PHX_INTRPT_SYNC_LOST       = -10003,   /* PHX Sync Lost event detected in callback */
            AS_EVENT_ERROR_ID_PHX_INTRPT_FRAME_LOST      = -10004    /* PHX Frame Lost event detected in callback */
         };
         typedef int32_t AS_EVENT_ERROR_ID;

         enum AS_EVENT_TYPE_LIST
         {
            AS_EVENT_MODULE = GenICam_GenTL::Client::EVENT_CUSTOM_ID + 1,   /* Corresponds to GenTL v1.5 EVENT_MODULE event type */
         };
         typedef int32_t AS_EVENT_TYPE_ID;

         /* ID values for the AS_EVENT_MODULE event. Values that are not documented here are identical in their meaning to
          * values bearing the same name in AS_EVENT_ERROR_ID_LIST.
          * The event ID can be retrieved using the EventGetDataInfo() function.
          */
         enum AS_EVENT_MODULE_ID_LIST
         {
            AS_EVENT_MODULE_ID_PHX_INTRPT_TEST              = 0x8001,
            AS_EVENT_MODULE_ID_PHX_INTRPT_FIFO_OVERFLOW     = 0x8002,
            AS_EVENT_MODULE_ID_PHX_INTRPT_FRAME_LOST        = 0x8003,
            AS_EVENT_MODULE_ID_PHX_INTRPT_FRAME_START       = 0x8004,
            AS_EVENT_MODULE_ID_PHX_INTRPT_FRAME_END         = 0x8005,
            AS_EVENT_MODULE_ID_PHX_INTRPT_LINE_START        = 0x8006,
            AS_EVENT_MODULE_ID_PHX_INTRPT_LINE_END          = 0x8007,
            AS_EVENT_MODULE_ID_PHX_INTRPT_SYNC_LOST         = 0x8008,
            AS_EVENT_MODULE_ID_PHX_INTRPT_CAMERA_TRIGGER    = 0x8009,
            AS_EVENT_MODULE_ID_PHX_INTRPT_TIMERM1           = 0x800A,
            AS_EVENT_MODULE_ID_PHX_INTRPT_TIMERM2           = 0x800B
         };
         typedef int32_t AS_EVENT_MODULE_ID;

         /* Additional buffer memory types */
         enum AS_BUFFER_MEM_TYPE_LIST
         {
            AS_BUFFER_MEM_TYPE_VIRTUAL_ADDRESS  = 0,  /* User buffers are allocated in virtual memory */
            AS_BUFFER_MEM_TYPE_PHYSICAL_ADDRESS = 1,  /* User buffers are allocated in physical memory, using a single address and length */
            AS_BUFFER_MEM_TYPE_PHYSICAL_ARRAY   = 2   /* User buffers are allocated in physical memory, using a zero-terminated array of addresses and lengths */
         };
         typedef int32_t AS_BUFFER_MEM_TYPE;

         /* Structure defining a physical entry used to describe buffers allocated the AS_BUFFER_MEM_TYPE_PHYSICAL_ARRAY memory type */
         typedef struct S_AS_PHYS_BUFFER
         {
            uint64_t uiAddress;  /* Address of the physical entry */
            uint64_t uiLength;   /* Length in bytes of the physical entry */
         } AS_PHYS_BUFFER;

         enum AS_INTERFACE_INFO_CMD_LIST
         {
            INTERFACE_INFO_BOARDNB = GenICam_GenTL::Client::INTERFACE_INFO_CUSTOM_ID /* UINT32_T Value of PHX parameter PHX_BOARD_NUMBER for PHX_BOARD_VARIANT = PHX_BOARD_DIGITAL. */
         };
         typedef int32_t AS_INTERFACE_INFO_CMD;

         enum AS_DEVICE_INFO_CMD_LIST
         {
            DEVICE_INFO_CHANNELNB = GenICam_GenTL::Client::DEVICE_INFO_CUSTOM_ID /* UINT32_T Value of PHX parameter PHX_CHANNEL_NUMBER. */
         };
         typedef int32_t AS_DEVICE_INFO_CMD;

         enum DEVICE_ACCESS_FLAGS_AS_LIST
         {
            DEVICE_ACCESS_AS_REMOTEDEVICE_PORT_ONLY = GenICam_GenTL::Client::DEVICE_ACCESS_CUSTOM_ID + 1 /* Open the device in a way that only the remote device port will be accessible.
                                                                                                            Allows controlling the remove device whilst the other modules are controlled 
                                                                                                            in another way. */
         };
         typedef int32_t DEVICE_ACCESS_FLAGS_AS;


         /* Additional functions */
         GC_IMPORT_EXPORT GenICam_GenTL::Client::GC_ERROR GC_CALLTYPE DSAnnounceBufferEx(GenICam_GenTL::Client::DS_HANDLE hDataStream, void *pBuffer, size_t iSize, void *pPrivate, GenICam_GenTL::Client::BUFFER_HANDLE *phBuffer, AS_BUFFER_MEM_TYPE iBufferMemType);

         typedef GenICam_GenTL::Client::GC_ERROR (GC_CALLTYPE *PDSAnnounceBufferEx)(GenICam_GenTL::Client::DS_HANDLE hDataStream, void *pBuffer, size_t iSize, void *pPrivate, GenICam_GenTL::Client::BUFFER_HANDLE *phBuffer, AS_BUFFER_MEM_TYPE iBufferMemType);

#ifdef __cplusplus
      } /* end of namespace GenTL */
   } /* end of namespace ActiveSilicon */
} /* end of extern "C" */
#endif
#endif /* TLACTIVESILICON_H_ */
