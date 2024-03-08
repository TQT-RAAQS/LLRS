/****************************************************************************
(c) 2004-2011 by GenICam GenTL Subcommittee

License: This file is published under the license of the EMVA GenICam Standard Group.
A text file describing the legal terms is included in your installation as 'license.txt'.
If for some reason you are missing this file please contact the EMVA or visit the website
(http://www.genicam.org) for a full copy.

THIS SOFTWARE IS PROVIDED BY THE EMVA GENICAM STANDARD GROUP "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE EMVA GENICAM STANDARD  GROUP
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,  SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT  LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE  OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

******************************************************************************/

/*  \file     GenTL.h
 *  \brief    GenICam Transport Layer Client Interface
 *  \version  1.3
 *  \author   rst (SI), cbi (IDS), jb (Leutron Vision), tho (MVTec), mjo(Mathworks)
 *  \date     21.08.2011
 *
 *  \par Version history
 *  \li Version 0.1.0   First official version from the meeting in Pilsen
 *  \li Version 0.1.1   rst (SI) 0.4.160
 *                      - added _E_INTERFACE_INFO_CMD_LIST_T_ enum name
 *                      - added _E_DEVICE_ACCCESS_STATUS_TYPE_T_ to be used
 *                      with the /a IFGetDeviceInfo and DevGetInfo
 *                      - rename of the enum value DEVICE_INFO_ACCESSMODE to
 *                      DEVICE_INFO_ACCESS_STATUS which now refers to
 *                      _E_DEVICE_ACCCESS_STATUS_TYPE_T_
 *                      - added Timeout parameter to update interface list and
 *                      device list.
 *  \li Version 0.1.2   - change datatype of timeout parameter of
 *                      TLUpdateInterfaceList and IFUpdateDeviceList to
 *                      uint64_t to match with the timeout in the event object.
 *                      - changed all enums to have a typedef to uint32_t
 *                      with them to allow custom ids
 *                      - changed type of string constants to be char * instead
 *                      of gcstring
 *  \li Version 0.1.3 rst (SI), cbi (IDS) 0.4.163
 *                      - adjusted parameter names to be closer aligned with
 *                      the standard text
 *                      - changed typedefs for enums from uint32_t to int32_t
 *                      - removed default parameter
 *                      - added parameter name to DevGetPort function
 *  \li Version 0.1.4 jb (LV)
 *                      - fixes to align the file with standard text
 *                      - make the file self-contained, independent on GenApi
 *  \li Version 0.1.5 rst (SI) cbi (IDS) jb (LV) tho (MVTec)
 *                      - Adjust it for Linux
 *                      - Cosmetics
 *  \li Version 1.0   rst (SI) cbi (IDS) jb (LV) tho (MVTec)
 *                      - Adjust for Standard 1.0
 *                      - Make it plain C compliant
 *                      - Cosmetics
 *  \li Version 1.2   rst (SI) jb (LV) tho (MVTec)
 *                      - Adjust for Standard 1.2
 *                      - adjust packing
 *                      - Cosmetics
 *  \li Version 1.3   (Stemmer, Leutron, Matrix, MVTec, Mathworks)
 *                      - Adjust for Standard 1.3
 *                      - added chunk handling
 *                      - added MAC OSx
 *                      - Cosmetics
 */


#ifndef GC_TLI_CLIENT_H_
#define GC_TLI_CLIENT_H_ 1

#ifndef GC_USER_DEFINED_TYPES
/* The types should be the same as defined in GCTypes.h from GenApi. But in
 * case you do not have this header the necessary types are defined here. */
#  if defined(_WIN32)
#    if defined(_MSC_VER) && _MSC_VER >= 1600 /* VS2010 provides stdint.h */
#      include <stdint.h>
#    elif !defined _STDINT_H && !defined _STDINT
       /* stdint.h is usually not available under Windows */
       typedef unsigned char uint8_t;
       typedef __int32 int32_t;
       typedef unsigned __int32 uint32_t;
       typedef unsigned __int64 uint64_t;
#    endif
#  else
#    include <stdint.h>
#  endif

#  ifdef __cplusplus
     typedef bool bool8_t;
#  else
     typedef uint8_t bool8_t;
#  endif
#endif /* GC_DEFINE_TYPES */

#include <stddef.h>


/* Function declaration modifiers */
#if defined (_WIN32)
#  ifndef GCTLI_NO_DECLSPEC_STATEMENTS
#    ifdef GCTLIDLL
#      define GC_IMPORT_EXPORT __declspec(dllexport)
#    else
#      define GC_IMPORT_EXPORT __declspec(dllimport)
#    endif
#  else
#      define GC_IMPORT_EXPORT
#  endif /* #  ifndef GCTLI_NO_DECLSPEC_STATEMENTS */
#  ifndef _M_X64
#    define GC_CALLTYPE __stdcall
#  else
#    define GC_CALLTYPE /* default */
#  endif
#  ifndef EXTERN_C
#    define EXTERN_C extern "C"
#  endif

#elif defined (__GNUC__) && (__GNUC__ >= 4) && (defined (__linux__) || defined (__APPLE__))
#  define GC_IMPORT_EXPORT __attribute__((visibility("default")))
#  ifndef __x86_64
#    define GC_CALLTYPE __attribute__((stdcall))
#  else
#    define GC_CALLTYPE /* default */
#  endif
#  ifndef EXTERN_C
#    define EXTERN_C extern "C"
#  endif

#else
#  error Unknown platform, file needs adaption
#endif

#ifdef __cplusplus
extern "C" {
  namespace GenICam_GenTL {
    namespace Client {
#endif

      /* Errors */
      enum GC_ERROR_LIST
      {
        GC_ERR_SUCCESS             = 0,
        GC_ERR_ERROR               = -1001,
        GC_ERR_NOT_INITIALIZED     = -1002,
        GC_ERR_NOT_IMPLEMENTED     = -1003,
        GC_ERR_RESOURCE_IN_USE     = -1004,
        GC_ERR_ACCESS_DENIED       = -1005,
        GC_ERR_INVALID_HANDLE      = -1006,
        GC_ERR_INVALID_ID          = -1007,
        GC_ERR_NO_DATA             = -1008,
        GC_ERR_INVALID_PARAMETER   = -1009,
        GC_ERR_IO                  = -1010,
        GC_ERR_TIMEOUT             = -1011,
        GC_ERR_ABORT               = -1012, /* GenTL v1.1 */
        GC_ERR_INVALID_BUFFER      = -1013, /* GenTL v1.1 */
        GC_ERR_NOT_AVAILABLE       = -1014, /* GenTL v1.2 */
        GC_ERR_INVALID_ADDRESS     = -1015, /* GenTL v1.3 */

        GC_ERR_CUSTOM_ID           = -10000
      };
      typedef int32_t GC_ERROR;

#     ifndef GC_GENTL_HEADER_VERSION

#       define GenTLMajorVersion       1 /* defines the major version of the GenICam GenTL standard version this header is based on */
#       define GenTLMinorVersion       3 /* defines the minor version of the GenICam GenTL standard version this header is based on */
#       define GenTLSubMinorVersion    0 /* defines the sub minor version of the GenICam GenTL standard version this header is based on */

#       define GC_GENTL_HEADER_VERSION_CODE(major,minor,subminor) (((major)<<24)+((minor)<<16)+(subminor))
#       define GC_GENTL_HEADER_VERSION GC_GENTL_HEADER_VERSION_CODE(GenTLMajorVersion,GenTLMinorVersion,GenTLSubMinorVersion)

#     endif /* GC_GENTL_HEADER_VERSION */

#     ifndef GC_GENTL_DONT_USE_TYPE_DEFINITIONS
#     define TLTypeMixedName           "Mixed"    /* Type to use for several supported technologies */
#     define TLTypeCustomName          "Custom"   /* Type to use for custom technologies */
#     define TLTypeGEVName             "GEV"      /* Type to use for GigE Vision technology */
#     define TLTypeCLName              "CL"       /* Type to use for Camera Link technology */
#     define TLTypeIIDCName            "IIDC"     /* Type to use for IIDC 1394 technology */
#     define TLTypeUVCName             "UVC"      /* Type to use for USB video class devices */
#     define TLTypeCXPName             "CXP"      /* Type to use for CoaXPress, V1.3 */
#     define TLTypeCLHSName            "CLHS"     /* Type to use for Camera Link HS, V1.3 */
#     define TLTypeUSB3Name            "USB3"     /* Type to use for USB3 Vision Standard, V1.3 */
#     define TLTypeETHERNETName        "Ethernet" /* Type to use for Ethernet devices, V1.3 */
#     define TLTypePCIName             "PCI"      /* Type to use for PCI/PCIe devices, V1.3 */
#     endif  /* GC_GENTL_DONT_USE_TYPE_DEFINITIONS */

#     ifndef GC_GENTL_DONT_USE_MODULE_NAMES
#     define TLSystemModuleName        "TLSystem"     /* Name to identify a port belonging to the system module */
#     define TLInterfaceModuleName     "TLInterface"  /* Name to identify a port belonging to the interface module */
#     define TLDeviceModuleName        "TLDevice"     /* Name to identify a port belonging to the device module */
#     define TLDataStreamModuleName    "TLDataStream" /* Name to identify a port belonging to the data stream module */
#     define TLBufferModuleName        "TLBuffer"     /* Name to identify a port belonging to the buffer module */
#     define TLRemoteDeviceModuleName  "Device"       /* Name to identify a port belonging to the remote device module */
#     endif /* GC_GENTL_DONT_USE_MODULE_NAMES */

      /* Handles */
      typedef void *      TL_HANDLE;         /* Transport Layer handle, obtained through the TLOpen */
      typedef void *      IF_HANDLE;         /* Interface handle, obtained through ::TLOpenInterface */
      typedef void *      DEV_HANDLE;        /* Device Handle, obtained through the ::IFOpenDevice */
      typedef void *      DS_HANDLE;         /* Handle to an image stream object, obtained through DevOpenDataStream */
      typedef void *      PORT_HANDLE;       /* A Port handle is used to access the register space of a port */
                                             /*  a PORT_HANDLE can be one of the following TL_HANDLE, IF_HANDLE, */
                                             /*  DEV_HANDLE, handle to a device port, obtained through ::DevGetPort, */
                                             /*  DS_HANDLE, BUFFER_HANDLE */

      typedef void *      BUFFER_HANDLE;     /* BufferHandle, obtained through the ::DSAnnounceBuffer function */
      typedef void *      EVENTSRC_HANDLE;   /* A Event source handle is used to register a OS Event and to retrieve a GenTL event handle */
                                             /* a EVENTSRC_HANDLE can be on of the following TL_HANDLE, */
                                             /* IF_HANDLE, DEV_HANDLE, A handle to a device port, obtained through ::DevGetPort */
                                             /* DS_HANDLE, BUFFER_HANDLE */
      typedef void *      EVENT_HANDLE;      /* Event Handle */


      /* Defines the data type possible for the various Info functions. */
      enum INFO_DATATYPE_LIST
      {
        INFO_DATATYPE_UNKNOWN     = 0,        /* Unknown data type */
        INFO_DATATYPE_STRING      = 1,        /* NULL-terminated C string (ASCII encoded). */
        INFO_DATATYPE_STRINGLIST  = 2,        /* Concatenated INFO_DATATYPE_STRING list. End of list is signaled with an additional NULL. */
        INFO_DATATYPE_INT16       = 3,        /* Signed 16 bit integer. */
        INFO_DATATYPE_UINT16      = 4,        /* Unsigned 16 bit integer */
        INFO_DATATYPE_INT32       = 5,        /* Signed 32 bit integer */
        INFO_DATATYPE_UINT32      = 6,        /* Unsigned 32 bit integer */
        INFO_DATATYPE_INT64       = 7,        /* Signed 64 bit integer */
        INFO_DATATYPE_UINT64      = 8,        /* Unsigned 64 bit integer */
        INFO_DATATYPE_FLOAT64     = 9,        /* Signed 64 bit floating point number. */
        INFO_DATATYPE_PTR         = 10,       /* Pointer type (void*). Size is platform dependent (32 bit on 32 bit platforms). */
        INFO_DATATYPE_BOOL8       = 11,       /* Boolean value occupying 8 bit. 0 for false and anything for true. */
        INFO_DATATYPE_SIZET       = 12,       /* Platform dependent unsigned integer (32 bit on 32 bit platforms). */
        INFO_DATATYPE_BUFFER      = 13,       /* Like a INFO_DATATYPE_STRING but with arbitrary data and no NULL termination. */
        INFO_DATATYPE_PTRDIFF     = 14,       /* Platform dependent unsigned integer (32 bit on 32 bit platforms). Negative is invalid. GenTL v1.3 */

        INFO_DATATYPE_CUSTOM_ID   = 1000      /* Starting value for custom IDs. */
      };
      typedef int32_t INFO_DATATYPE;

      /* System module information commands for the GenICam::TL::Client::TLGetInfo and GenICam::TL::Client::GCGetInfo functions. */
      enum  TL_INFO_CMD_LIST
      {
        TL_INFO_ID              = 0,    /* STRING    Transport layer ID. */
        TL_INFO_VENDOR          = 1,    /* STRING    Transport layer vendor name. */
        TL_INFO_MODEL           = 2,    /* STRING    Transport layer model name. */
        TL_INFO_VERSION         = 3,    /* STRING    Transport layer revision. */
        TL_INFO_TLTYPE          = 4,    /* STRING    Transport layer technology that is supported. */
        TL_INFO_NAME            = 5,    /* STRING    File name including extension of the library. */
        TL_INFO_PATHNAME        = 6,    /* STRING    Full path including file name and extension of the library. */
        TL_INFO_DISPLAYNAME     = 7,    /* STRING    User readable name of the device. If this is not defined in the device this should be VENDOR MODEL (ID). */

        TL_INFO_CUSTOM_ID       = 1000  /* Starting value for custom IDs. */
      };
      typedef int32_t TL_INFO_CMD;

      /* This enumeration defines commands to retrieve information with the GenICam::TL::Client::IFGetInfo function from the Interface module. */
      enum INTERFACE_INFO_CMD_LIST
      {
        INTERFACE_INFO_ID              = 0,     /* STRING     Unique ID of the interface. */
        INTERFACE_INFO_DISPLAYNAME     = 1,     /* STRING     User readable name of the interface. */
        INTERFACE_INFO_TLTYPE          = 2,     /* STRING     Transport layer technology that is supported. */

        INTERFACE_INFO_CUSTOM_ID       = 1000   /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t INTERFACE_INFO_CMD;

      /* This enumeration defines flags of how a device is to be opened with the GenICam::TL::Client::IFOpenDevice function. */
      enum DEVICE_ACCESS_FLAGS_LIST
      {
        DEVICE_ACCESS_UNKNOWN   = 0,         /* Not used in a command. Can be used to initialize a variable to query that information. */
        DEVICE_ACCESS_NONE      = 1,         /* This either means that the device is not open because it was not opened before or the access to it was denied. */
        DEVICE_ACCESS_READONLY  = 2,         /* Open the device read only. All Port functions can only read from the device. */
        DEVICE_ACCESS_CONTROL   = 3,         /* Open the device in a way that other hosts/processes can have read only access to the device. Device access level is read/write for this process. */
        DEVICE_ACCESS_EXCLUSIVE = 4,         /* Open the device in a way that only this host/process can have access to the device. Device access level is read/write for this process. */

        DEVICE_ACCESS_CUSTOM_ID = 1000       /*  Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t DEVICE_ACCESS_FLAGS;

      /* This enumeration defines values for the accessibility of the device to be returned in the GenICam::TL::Client::DevGetInfo function on a device handle. */
      enum DEVICE_ACCESS_STATUS_LIST
      {
        DEVICE_ACCESS_STATUS_UNKNOWN   = 0,   /* The device accessibility is not known. */
        DEVICE_ACCESS_STATUS_READWRITE = 1,   /* The device is available for read/write access. */
        DEVICE_ACCESS_STATUS_READONLY  = 2,   /* The device is available for read only access. */
        DEVICE_ACCESS_STATUS_NOACCESS  = 3,   /* The device is not accessible. */

        DEVICE_ACCESS_STATUS_CUSTOM_ID = 1000 /* Starting value for custom IDs. */
      };
      typedef int32_t DEVICE_ACCESS_STATUS;

      /* This enumeration defines commands to retrieve information with the GenICam::TL::Client::DevGetInfo function on a device handle. */
      enum DEVICE_INFO_CMD_LIST
      {
        DEVICE_INFO_ID              = 0,      /* STRING      Unique ID of the device. */
        DEVICE_INFO_VENDOR          = 1,      /* STRING      Device vendor name. */
        DEVICE_INFO_MODEL           = 2,      /* STRING      Device model name. */
        DEVICE_INFO_TLTYPE          = 3,      /* STRING      Transport layer technology that is supported. */
        DEVICE_INFO_DISPLAYNAME     = 4,      /* STRING      String containing a display name for the device ( including a unique id ) */
        DEVICE_INFO_ACCESS_STATUS   = 5,      /* INT32       Gets the access mode the GenTL Producer has on the opened device. (DEVICE_ACCESS_STATUS enumeration value). */

        DEVICE_INFO_CUSTOM_ID       = 1000    /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t DEVICE_INFO_CMD;

      /* This enumeration defines special stop flags for the acquisition engine. The function used is GenICam::TL::Client::DSStopAcquisition. */
      enum ACQ_STOP_FLAGS_LIST
      {
        ACQ_STOP_FLAGS_DEFAULT   = 0,         /* Stop the acquisition engine when the currently running tasks like filling a buffer are completed (default behavior). */
        ACQ_STOP_FLAG_KILL       = 1,         /* Stop the acquisition engine immediately and leave buffers currently being filled in the Input Buffer Pool. */

        ACQ_STOP_CUSTOM_ID       = 1000       /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t ACQ_STOP_FLAGS;

      /* This enumeration defines special start flags for the acquisition engine. The function used is GenICam::TL::Client::DSStartAcquisition. */
      enum ACQ_START_FLAGS_LIST
      {
        ACQ_START_FLAGS_DEFAULT     = 0,      /* Default behavior. */

        ACQ_START_FLAGS_CUSTOM_ID   = 1000    /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t ACQ_START_FLAGS;

      /* This enumeration commands from which to which queue/pool buffers are flushed with the GenICam::TL::Client::DSFlushQueue function. */
      enum ACQ_QUEUE_TYPE_LIST
      {
        ACQ_QUEUE_INPUT_TO_OUTPUT           = 0,    /* Flushes the input pool to the output queue and if necessary adds entries in the New Buffer event data queue. */
        ACQ_QUEUE_OUTPUT_DISCARD            = 1,    /* Discards all buffers in the output queue and if necessary remove the entries from the event data queue. */
        ACQ_QUEUE_ALL_TO_INPUT              = 2,    /* Puts all buffers in the input pool. Even those in the output queue and discard entries in the event data queue. */
        ACQ_QUEUE_UNQUEUED_TO_INPUT         = 3,    /* Puts all buffers that are not in the input pool or the output queue in the input pool. */
        ACQ_QUEUE_ALL_DISCARD               = 4,    /* Discards all buffers in the input pool and output queue. */

        ACQ_QUEUE_CUSTOM_ID                 = 1000  /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t ACQ_QUEUE_TYPE;

      /* This enumeration defines commands to retrieve information with the GenICam::TL::Client::DSGetInfo function on a data stream handle */
      enum STREAM_INFO_CMD_LIST
      {
        STREAM_INFO_ID                         =  0,   /* STRING     Unique ID of the data stream. */
        STREAM_INFO_NUM_DELIVERED              =  1,   /* UINT64     Number of acquired frames since last acquisition start. */
        STREAM_INFO_NUM_UNDERRUN               =  2,   /* UINT64     Number of lost frames due to queue underrun. */
        STREAM_INFO_NUM_ANNOUNCED              =  3,   /* SIZET      Number of announced buffers. */
        STREAM_INFO_NUM_QUEUED                 =  4,   /* SIZET      Number of buffers in the input pool. */
        STREAM_INFO_NUM_AWAIT_DELIVERY         =  5,   /* SIZET      Number of buffers in the output queue. */
        STREAM_INFO_NUM_STARTED                =  6,   /* UINT64     Number of frames started in the acquisition engine. */
        STREAM_INFO_PAYLOAD_SIZE               =  7,   /* SIZET      Size of the expected data in bytes. */
        STREAM_INFO_IS_GRABBING                =  8,   /* BOOL8      Flag indicating whether the acquisition engine is started or not. */
        STREAM_INFO_DEFINES_PAYLOADSIZE        =  9,   /* BOOL8      Flag that indicated that this data stream defines a payload size independent from the remote device. */
        STREAM_INFO_TLTYPE                     = 10,   /* STRING     Transport layer technology that is supported. */
        STREAM_INFO_NUM_CHUNKS_MAX             = 11,   /* SIZET      Max number of chunks in a buffer, if known. GenTL v1.3 */
        STREAM_INFO_NUM_ANNOUNCE_MIN           = 12,   /* SIZET      Min number of buffers to announce before acq can start, if known. GenTL v1.3 */
        STREAM_INFO_BUF_ALIGNMENT              = 13,   /* SIZET      Buffer alignment in bytes. GenTL v1.3 */

        STREAM_INFO_CUSTOM_ID                  = 1000  /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t STREAM_INFO_CMD;

      /* This enumeration defines commands to retrieve information with the GenICam::TL::Client::DSGetBufferInfo function on a buffer handle. */
      enum  BUFFER_INFO_CMD_LIST
      {
        BUFFER_INFO_BASE                            = 0,  /* PTR        Base address of the buffer memory. */
        BUFFER_INFO_SIZE                            = 1,  /* SIZET      Size of the buffer in bytes. */
        BUFFER_INFO_USER_PTR                        = 2,  /* PTR        Private data pointer of the GenTL Consumer. */
        BUFFER_INFO_TIMESTAMP                       = 3,  /* UINT64     Timestamp the buffer was acquired. */
        BUFFER_INFO_NEW_DATA                        = 4,  /* BOOL8      Flag to indicate that the buffer contains new data since the last call. */
        BUFFER_INFO_IS_QUEUED                       = 5,  /* BOOL8      Flag to indicate if the buffer is in the input pool or output queue. */
        BUFFER_INFO_IS_ACQUIRING                    = 6,  /* BOOL8      Flag to indicate that the buffer is currently being filled with data. */
        BUFFER_INFO_IS_INCOMPLETE                   = 7,  /* BOOL8      Flag to indicate that a buffer was filled but an error occurred during that process. */
        BUFFER_INFO_TLTYPE                          = 8,  /* STRING     Transport layer technology that is supported. */
        BUFFER_INFO_SIZE_FILLED                     = 9,  /* SIZET      Number of bytes written into the buffer last time it has been filled. This value is reset to 0 when the buffer is placed into the Input Buffer Pool. */
        BUFFER_INFO_WIDTH                           = 10, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_HEIGHT                          = 11, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_XOFFSET                         = 12, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_YOFFSET                         = 13, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_XPADDING                        = 14, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_YPADDING                        = 15, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_FRAMEID                         = 16, /* UINT64     GenTL v1.2 */
        BUFFER_INFO_IMAGEPRESENT                    = 17, /* BOOL       GenTL v1.2 */
        BUFFER_INFO_IMAGEOFFSET                     = 18, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_PAYLOADTYPE                     = 19, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_PIXELFORMAT                     = 20, /* UINT64     GenTL v1.2 */
        BUFFER_INFO_PIXELFORMAT_NAMESPACE           = 21, /* UINT64     GenTL v1.2 */
        BUFFER_INFO_DELIVERED_IMAGEHEIGHT           = 22, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_DELIVERED_CHUNKPAYLOADSIZE      = 23, /* SIZET      GenTL v1.2 */
        BUFFER_INFO_CHUNKLAYOUTID                   = 24, /* UINT64     GenTL v1.2 */
        BUFFER_INFO_FILENAME                        = 25, /* STRING     GenTL v1.2 */

        BUFFER_INFO_CUSTOM_ID                       = 1000 /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t BUFFER_INFO_CMD;

      /* Enumeration of TLType dependent payload types. Introduced in GenTL v1.2 */
      enum PAYLOADTYPE_INFO_IDS
      {
        PAYLOAD_TYPE_UNKNOWN     = 0,   /* GenTL v1.2 */
        PAYLOAD_TYPE_IMAGE       = 1,   /* GenTL v1.2 */
        PAYLOAD_TYPE_RAW_DATA    = 2,   /* GenTL v1.2 */
        PAYLOAD_TYPE_FILE        = 3,   /* GenTL v1.2 */
        PAYLOAD_TYPE_CHUNK_DATA  = 4,   /* GenTL v1.2 */

        PAYLOAD_TYPE_CUSTOM_ID   = 1000
      };
      typedef int32_t PAYLOADTYPE_INFO_ID;

      /* Enumeration of TLType dependent pixel format namespaces introduced GenTL v1.2 */
      enum PIXELFORMAT_NAMESPACE_IDS
      {
        PIXELFORMAT_NAMESPACE_UNKNOWN   = 0,   /* GenTL v1.2 */
        PIXELFORMAT_NAMESPACE_GEV       = 1,   /* GenTL v1.2 */
        PIXELFORMAT_NAMESPACE_IIDC      = 2,   /* GenTL v1.2 */

        PIXELFORMAT_NAMESPACE_CUSTOM_ID = 1000
      };
      typedef int32_t PIXELFORMAT_NAMESPACE_ID;   /* GenTL v1.2 */

      /* This enumeration defines commands to retrieve information with the GenICam::TL::Client::GCGetPortInfo function on a module or remote device handle. */
      enum PORT_INFO_CMD_LIST
      {
        PORT_INFO_ID              = 0,       /* STRING     Unique ID of the port. */
        PORT_INFO_VENDOR          = 1,       /* STRING     Port vendor name. */
        PORT_INFO_MODEL           = 2,       /* STRING     Port model name. */
        PORT_INFO_TLTYPE          = 3,       /* STRING     Transport layer technology that is supported. */
        PORT_INFO_MODULE          = 4,       /* STRING     GenTL Module the port refers to. */
        PORT_INFO_LITTLE_ENDIAN   = 5,       /* BOOL8      Flag indicating that the port data is little endian. */
        PORT_INFO_BIG_ENDIAN      = 6,       /* BOOL8      Flag indicating that the port data is big endian. */
        PORT_INFO_ACCESS_READ     = 7,       /* BOOL8      Port has read access. */
        PORT_INFO_ACCESS_WRITE    = 8,       /* BOOL8      Port has write access. */
        PORT_INFO_ACCESS_NA       = 9,       /* BOOL8      Port in not accessible. */
        PORT_INFO_ACCESS_NI       = 10,      /* BOOL8      Port is not implemented. */
        PORT_INFO_VERSION         = 11,      /* STRING     Revision of the port. */
        PORT_INFO_PORTNAME        = 12,      /* STRING     Name of the port as referenced in the XML description. */

        PORT_INFO_CUSTOM_ID       = 1000     /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t PORT_INFO_CMD;

      /* This enumeration defines commands to retrieve information with the GenICam::TL::Client::GCGetPortURLInfo
         function on a module or remote device handle. Introduced in GenTL v1.1 */
      enum URL_INFO_CMD_LIST
      {
        URL_INFO_URL               = 0,       /* STRING     URL as defined in chapter 4.1.2 GenTL v1.1  */
        URL_INFO_SCHEMA_VER_MAJOR  = 1,       /* INT32      Major version of the schema this URL refers to. GenTL v1.1 */
        URL_INFO_SCHEMA_VER_MINOR  = 2,       /* INT32      Minor version of the schema this URL refers to. GenTL v1.1 */
        URL_INFO_FILE_VER_MAJOR    = 3,       /* INT32      Major version of the XML-file this URL refers to. GenTL v1.1 */
        URL_INFO_FILE_VER_MINOR    = 4,       /* INT32      Minor version of the XML-file this URL refers to. GenTL v1.1 */
        URL_INFO_FILE_VER_SUBMINOR = 5,       /* INT32      Subminor version of the XML-file this URL refers to. GenTL v1.1 */

        URL_INFO_CUSTOM_ID         = 1000     /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t URL_INFO_CMD;  /* GenTL v1.1 */

      /* Known event types that can be registered on certain modules with the GenICam::TL::Client::GCRegisterEvent function. */
      enum EVENT_TYPE_LIST
      {
        EVENT_ERROR               = 0,     /* Notification on module errors. */
        EVENT_NEW_BUFFER          = 1,     /* Notification on newly filled buffers. */
        EVENT_FEATURE_INVALIDATE  = 2,     /* Notification if a feature was changed by the GenTL Producer library and thus needs to be invalidated in the GenICam GenApi instance using the module. */
        EVENT_FEATURE_CHANGE      = 3,     /* Notification if the GenTL Producer library wants to manually set a feature in the GenICam GenApi instance using the module. */
        EVENT_FEATURE_DEVEVENT    = 4,     /* Notification if the GenTL Producer wants to inform the GenICam GenApi instance of the remote device that a GenApi compatible event was fired. */

        EVENT_CUSTOM_ID           = 1000   /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t  EVENT_TYPE;

      /* Event info command */
      enum EVENT_INFO_CMD_LIST
      {
        EVENT_EVENT_TYPE            = 0,    /* INT32      The event type of the event handle (EVENT_TYPE enum value). */
        EVENT_NUM_IN_QUEUE          = 1,    /* SIZET      Number of events in the event data queue. */
        EVENT_NUM_FIRED             = 2,    /* UINT64     Number of events that were fired since the creation of the module. */
        EVENT_SIZE_MAX              = 3,    /* SIZET      Max size of data carried with an event in bytes. GenTL v1.2 */
        EVENT_INFO_DATA_SIZE_MAX    = 4,    /* SIZET      Max size of data provided through EventGetDataInfo in bytes. GenTL v1.2 */

        EVENT_INFO_CUSTOM_ID        = 1000  /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t EVENT_INFO_CMD;

      /* Event data info command */
      enum EVENT_DATA_INFO_CMD_LIST
      {
        EVENT_DATA_ID           = 0,      /* Event specific    Unique Event ID (String or Number)*/
        EVENT_DATA_VALUE        = 1,      /* Event specific Data */
        EVENT_DATA_NUMID        = 2,      /* UNINT64   Numeric representation of the unique Event ID, GenTL v1.3. */

        EVENT_DATA_CUSTOM_ID    = 1000    /* Starting value for GenTL Producer custom IDs. */
      };
      typedef int32_t EVENT_DATA_INFO_CMD;

      /* Structure of the data returned from a signaled "New Buffer" event. */
#     pragma pack (push, 1)
      typedef struct S_EVENT_NEW_BUFFER
      {
        BUFFER_HANDLE   BufferHandle;        /* Buffer handle which contains new data. */
        void*           pUserPointer;        /* User pointer provided at announcement of the buffer. */
      } EVENT_NEW_BUFFER_DATA;
#     pragma pack (pop)

      /* Structure to be use with GCWritePortStacked and GCReadPortStacked. */
#     pragma pack (push, 1)
      typedef struct S_PORT_REGISTER_STACK_ENTRY
      {
        uint64_t   Address;        /* Address of the register. */
        void*      pBuffer;        /* Pointer to the buffer containing the data. */
        size_t     Size;           /* Number of bytes to read write. */
      } PORT_REGISTER_STACK_ENTRY;
#     pragma pack (pop)

#     pragma pack (push, 1)
      /* Structure carrying information about a single chunk in the buffer, V1.3 */
      typedef struct S_SINGLE_CHUNK_DATA
      {
        uint64_t ChunkID;       /* Numeric representation of ChunkID */
        ptrdiff_t ChunkOffset;  /* Chunk offset in the buffer */
        size_t ChunkLength;     /* Size of the chunk data */
      } SINGLE_CHUNK_DATA;
#     pragma pack (pop)


      /* C API Interface Functions */
#     define GC_API GC_IMPORT_EXPORT GC_ERROR GC_CALLTYPE
      GC_API GCGetInfo               ( TL_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );

      GC_API GCGetLastError          ( GC_ERROR *piErrorCode, char *sErrText, size_t *piSize );

      GC_API GCInitLib               ( void );
      GC_API GCCloseLib              ( void );

      GC_API GCReadPort              ( PORT_HANDLE hPort, uint64_t iAddress, void *pBuffer, size_t *piSize );
      GC_API GCWritePort             ( PORT_HANDLE hPort, uint64_t iAddress, const void *pBuffer, size_t *piSize );
      GC_API GCGetPortURL            ( PORT_HANDLE hPort, char *sURL, size_t *piSize );

      GC_API GCGetPortInfo           ( PORT_HANDLE hPort, PORT_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );

      GC_API GCRegisterEvent         ( EVENTSRC_HANDLE hEventSrc, EVENT_TYPE iEventID, EVENT_HANDLE *phEvent );
      GC_API GCUnregisterEvent       ( EVENTSRC_HANDLE hEventSrc, EVENT_TYPE iEventID );

      GC_API EventGetData            ( EVENT_HANDLE hEvent, void *pBuffer, size_t *piSize, uint64_t iTimeout );
      GC_API EventGetDataInfo        ( EVENT_HANDLE hEvent, const void *pInBuffer, size_t iInSize, EVENT_DATA_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pOutBuffer, size_t *piOutSize );
      GC_API EventGetInfo            ( EVENT_HANDLE hEvent, EVENT_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API EventFlush              ( EVENT_HANDLE hEvent );
      GC_API EventKill               ( EVENT_HANDLE hEvent );

      GC_API TLOpen                  ( TL_HANDLE *phTL );
      GC_API TLClose                 ( TL_HANDLE hTL );
      GC_API TLGetInfo               ( TL_HANDLE hTL, TL_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );

      GC_API TLGetNumInterfaces      ( TL_HANDLE hTL, uint32_t *piNumIfaces );
      GC_API TLGetInterfaceID        ( TL_HANDLE hTL, uint32_t iIndex,  char *sID, size_t *piSize );
      GC_API TLGetInterfaceInfo      ( TL_HANDLE hTL, const char *sIfaceID, INTERFACE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API TLOpenInterface         ( TL_HANDLE hTL, const char *sIfaceID, IF_HANDLE *phIface );
      GC_API TLUpdateInterfaceList   ( TL_HANDLE hTL, bool8_t *pbChanged, uint64_t iTimeout );

      GC_API IFClose                 ( IF_HANDLE hIface );
      GC_API IFGetInfo               ( IF_HANDLE hIface, INTERFACE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );

      GC_API IFGetNumDevices         ( IF_HANDLE hIface, uint32_t *piNumDevices );
      GC_API IFGetDeviceID           ( IF_HANDLE hIface, uint32_t iIndex, char *sIDeviceID, size_t *piSize );
      GC_API IFUpdateDeviceList      ( IF_HANDLE hIface, bool8_t *pbChanged, uint64_t iTimeout );
      GC_API IFGetDeviceInfo         ( IF_HANDLE hIface, const char *sDeviceID, DEVICE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API IFOpenDevice            ( IF_HANDLE hIface, const char *sDeviceID, DEVICE_ACCESS_FLAGS iOpenFlags, DEV_HANDLE *phDevice );

      GC_API DevGetPort              ( DEV_HANDLE hDevice, PORT_HANDLE *phRemoteDevice );
      GC_API DevGetNumDataStreams    ( DEV_HANDLE hDevice, uint32_t *piNumDataStreams );
      GC_API DevGetDataStreamID      ( DEV_HANDLE hDevice, uint32_t iIndex, char *sDataStreamID, size_t *piSize );
      GC_API DevOpenDataStream       ( DEV_HANDLE hDevice, const char *sDataStreamID, DS_HANDLE *phDataStream );
      GC_API DevGetInfo              ( DEV_HANDLE hDevice, DEVICE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API DevClose                ( DEV_HANDLE hDevice );

      GC_API DSAnnounceBuffer        ( DS_HANDLE hDataStream, void *pBuffer, size_t iSize, void *pPrivate, BUFFER_HANDLE *phBuffer );
      GC_API DSAllocAndAnnounceBuffer( DS_HANDLE hDataStream, size_t iSize, void *pPrivate, BUFFER_HANDLE *phBuffer );
      GC_API DSFlushQueue            ( DS_HANDLE hDataStream, ACQ_QUEUE_TYPE iOperation );
      GC_API DSStartAcquisition      ( DS_HANDLE hDataStream, ACQ_START_FLAGS iStartFlags, uint64_t iNumToAcquire );
      GC_API DSStopAcquisition       ( DS_HANDLE hDataStream, ACQ_STOP_FLAGS iStopFlags );
      GC_API DSGetInfo               ( DS_HANDLE hDataStream, STREAM_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API DSGetBufferID           ( DS_HANDLE hDataStream, uint32_t iIndex, BUFFER_HANDLE *phBuffer );
      GC_API DSClose                 ( DS_HANDLE hDataStream );

      GC_API DSRevokeBuffer          ( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer, void **pBuffer, void **pPrivate );
      GC_API DSQueueBuffer           ( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer );
      GC_API DSGetBufferInfo         ( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer, BUFFER_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );

      /* GenTL v1.1 */
      GC_API GCGetNumPortURLs        ( PORT_HANDLE hPort, uint32_t *piNumURLs );
      GC_API GCGetPortURLInfo        ( PORT_HANDLE hPort, uint32_t iURLIndex, URL_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API GCReadPortStacked       ( PORT_HANDLE hPort, PORT_REGISTER_STACK_ENTRY *pEntries, size_t *piNumEntries );
      GC_API GCWritePortStacked      ( PORT_HANDLE hPort, PORT_REGISTER_STACK_ENTRY *pEntries, size_t *piNumEntries );

      /* GenTL v1.3 */
      GC_API DSGetBufferChunkData    ( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer, SINGLE_CHUNK_DATA *pChunkData, size_t *piNumChunks );

      /* typedefs for dynamic loading */
#     define GC_API_P(function) typedef GC_ERROR( GC_CALLTYPE *function )
      GC_API_P(PGCGetInfo               )( TL_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PGCGetLastError          )( GC_ERROR *piErrorCode, char *sErrText, size_t *piSize );
      GC_API_P(PGCInitLib               )( void );
      GC_API_P(PGCCloseLib              )( void );
      GC_API_P(PGCReadPort              )( PORT_HANDLE hPort, uint64_t iAddress, void *pBuffer, size_t *piSize );
      GC_API_P(PGCWritePort             )( PORT_HANDLE hPort, uint64_t iAddress, const void *pBuffer, size_t *piSize );
      GC_API_P(PGCGetPortURL            )( PORT_HANDLE hPort, char *sURL, size_t *piSize );
      GC_API_P(PGCGetPortInfo           )( PORT_HANDLE hPort, PORT_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );

      GC_API_P(PGCRegisterEvent         )( EVENTSRC_HANDLE hEventSrc, EVENT_TYPE iEventID, EVENT_HANDLE *phEvent );
      GC_API_P(PGCUnregisterEvent       )( EVENTSRC_HANDLE hEventSrc, EVENT_TYPE iEventID );
      GC_API_P(PEventGetData            )( EVENT_HANDLE hEvent, void *pBuffer, size_t *piSize, uint64_t iTimeout );
      GC_API_P(PEventGetDataInfo        )( EVENT_HANDLE hEvent, const void *pInBuffer, size_t iInSize, EVENT_DATA_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pOutBuffer, size_t *piOutSize );
      GC_API_P(PEventGetInfo            )( EVENT_HANDLE hEvent, EVENT_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PEventFlush              )( EVENT_HANDLE hEvent );
      GC_API_P(PEventKill               )( EVENT_HANDLE hEvent );
      GC_API_P(PTLOpen                  )( TL_HANDLE *phTL );
      GC_API_P(PTLClose                 )( TL_HANDLE hTL );
      GC_API_P(PTLGetInfo               )( TL_HANDLE hTL, TL_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PTLGetNumInterfaces      )( TL_HANDLE hTL, uint32_t *piNumIfaces );
      GC_API_P(PTLGetInterfaceID        )( TL_HANDLE hTL, uint32_t iIndex, char *sID, size_t *piSize );
      GC_API_P(PTLGetInterfaceInfo      )( TL_HANDLE hTL, const char *sIfaceID, INTERFACE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PTLOpenInterface         )( TL_HANDLE hTL, const char *sIfaceID, IF_HANDLE *phIface );
      GC_API_P(PTLUpdateInterfaceList   )( TL_HANDLE hTL, bool8_t *pbChanged, uint64_t iTimeout );
      GC_API_P(PIFClose                 )( IF_HANDLE hIface );
      GC_API_P(PIFGetInfo               )( IF_HANDLE hIface, INTERFACE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PIFGetNumDevices         )( IF_HANDLE hIface, uint32_t *piNumDevices );
      GC_API_P(PIFGetDeviceID           )( IF_HANDLE hIface, uint32_t iIndex, char *sIDeviceID, size_t *piSize );
      GC_API_P(PIFUpdateDeviceList      )( IF_HANDLE hIface, bool8_t *pbChanged, uint64_t iTimeout );
      GC_API_P(PIFGetDeviceInfo         )( IF_HANDLE hIface, const char *sDeviceID, DEVICE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PIFOpenDevice            )( IF_HANDLE hIface, const char *sDeviceID, DEVICE_ACCESS_FLAGS iOpenFlags, DEV_HANDLE *phDevice );
      GC_API_P(PDevGetPort              )( DEV_HANDLE hDevice, PORT_HANDLE *phRemoteDevice );
      GC_API_P(PDevGetNumDataStreams    )( DEV_HANDLE hDevice, uint32_t *piNumDataStreams );
      GC_API_P(PDevGetDataStreamID      )( DEV_HANDLE hDevice, uint32_t iIndex, char *sDataStreamID, size_t *piSize );
      GC_API_P(PDevOpenDataStream       )( DEV_HANDLE hDevice, const char *sDataStreamID, DS_HANDLE *phDataStream );
      GC_API_P(PDevGetInfo              )( DEV_HANDLE hDevice, DEVICE_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PDevClose                )( DEV_HANDLE hDevice );
      GC_API_P(PDSAnnounceBuffer        )( DS_HANDLE hDataStream, void *pBuffer, size_t iSize, void *pPrivate, BUFFER_HANDLE *phBuffer );
      GC_API_P(PDSAllocAndAnnounceBuffer)( DS_HANDLE hDataStream, size_t iSize, void *pPrivate, BUFFER_HANDLE *phBuffer );
      GC_API_P(PDSFlushQueue            )( DS_HANDLE hDataStream, ACQ_QUEUE_TYPE iOperation );
      GC_API_P(PDSStartAcquisition      )( DS_HANDLE hDataStream, ACQ_START_FLAGS iStartFlags, uint64_t iNumToAcquire );
      GC_API_P(PDSStopAcquisition       )( DS_HANDLE hDataStream, ACQ_STOP_FLAGS iStopFlags );
      GC_API_P(PDSGetInfo               )( DS_HANDLE hDataStream, STREAM_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PDSGetBufferID           )( DS_HANDLE hDataStream, uint32_t iIndex, BUFFER_HANDLE *phBuffer );
      GC_API_P(PDSClose                 )( DS_HANDLE hDataStream );
      GC_API_P(PDSRevokeBuffer          )( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer, void **pBuffer, void **pPrivate );
      GC_API_P(PDSQueueBuffer           )( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer );
      GC_API_P(PDSGetBufferInfo         )( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer, BUFFER_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );

      /* GenTL v1.1 */
      GC_API_P(PGCGetNumPortURLs        )( PORT_HANDLE hPort, uint32_t *iNumURLs );
      GC_API_P(PGCGetPortURLInfo        )( PORT_HANDLE hPort, uint32_t iURLIndex, URL_INFO_CMD iInfoCmd, INFO_DATATYPE *piType, void *pBuffer, size_t *piSize );
      GC_API_P(PGCReadPortStacked       )( PORT_HANDLE hPort, PORT_REGISTER_STACK_ENTRY *pEntries, size_t *piNumEntries );
      GC_API_P(PGCWritePortStacked      )( PORT_HANDLE hPort, PORT_REGISTER_STACK_ENTRY *pEntries, size_t *piNumEntries );

      /* GenTL v1.3 */
      GC_API_P(PDSGetBufferChunkData    )( DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer, SINGLE_CHUNK_DATA *pChunkData, size_t *piNumChunks );

#ifdef __cplusplus
    } /* end of namespace Client */
  } /* end of namespace GenICam */
} /* end of extern "C" */
#endif
#endif /* GC_TLI_CLIENT_H_ */
