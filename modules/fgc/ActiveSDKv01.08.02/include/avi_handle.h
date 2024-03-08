/* Private interface to ASL .avi handling libraries.
 *
 * Authors:          Graham Wright
 *
 * System:           ANSI C
 *
 * See ASL_Avi_Lib.h for version information
 *
 * Copyright (C) Active Silicon Limited 2006-2010
 *
 */

#define _ASL_AVI_TYPE         FCC("AVI ")    /* Standard code for .avi files */
#define _ASL_AUD_CHUNK        FCC("00wb")    /* Audio data chunk type */
#define _ASL_BMP_CHUNK        FCC("01db")    /* Video uncompressed bitmap data chunk type */
#define _ASL_MJPG_CHUNK       FCC("01dc")    /* Video MPJPEG data chunk type */
#define _ASL_AUD_INDX         FCC("ix00")    /* Audio index marker */
#define _ASL_VID_INDX         FCC("ix01")    /* Video index marker */
#define _ASL_CHUNK_LIMIT      8192           /* Maximum number of movi data chunks, each of which is up to roughly _ASL_CHUNK_SIZE (1GB) in size & has one index.*/
#define _ASL_CHUNK_SIZE       1073741312     /* 1GB limit (rather less in fact, to be on the safe side), for any movi chunk. This is necssary because of restrictions in many players. */ 
#define _ASL_CHUNK_FRAMES     20000          /* Maximum number of frames within any given chunk */
#define _ASL_BUFFER_FRAMES    100000         /* Maximum number of individual frames that can be written to RAM buffer */
#define _ASL_MIN_BUFFER_SIZE  104857600      /* Minimum size of memory buffer, excluding frame index, for buffered writes */
#define _ASL_INIT             542390716      /* Magic number which when set in a global tells other functions that the function has been initialiazed */
#define _ASL_MJPG             "MJPEG"        /* Strings hidden in the odml section at the end of the main header to define extended ASL types such as 48 bit */
#define _ASL_MJPGi            "Interlaced MJPEG"
#define _ASL_BGR24            "24 BIT BGR"
#define _ASL_BGR24i           "Interlaced 24 BIT BGR"
#define _ASL_BGR32            "32 BIT BGR CUSTOM TYPE"
#define _ASL_BGR48            "48 BIT BGR CUSTOM TYPE"
#define _ASL_MONO8            "8 BIT MONO"
#define _ASL_MONO16           "16 BIT MONO CUSTOM TYPE"
#define _ASL_UNDEFINED        "UNDEFINED"

/* Error handling macros */
#define _AVI_INVALID_ARGUMENT(Err)     if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Invalid Function Argument", AVI_INVALID_ARGUMENT);}\
                                       Err = AVI_INVALID_ARGUMENT;\
                                       goto Exit
#define _AVI_INVALID_HANDLE(Err)       if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Invalid Handle", AVI_INVALID_HANDLE);}\
                                       Err = AVI_INVALID_HANDLE;\
                                       goto Exit
#define _AVI_NOT_INIT(Err)             if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Avi Library Not Initialized", AVI_NOT_INIT);}\
                                       Err = AVI_NOT_INIT;\
                                       goto Exit
#define _AVI_NULL_POINTER(Err)         if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Null Pointer Passed to Function", AVI_NULL_POINTER);}\
                                       Err = AVI_NULL_POINTER;\
                                       goto Exit
#define _AVI_MEM_ALLOC(Err)            if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Memory Allocation Error", AVI_MEM_ALLOC);}\
                                       Err = AVI_MEM_ALLOC;\
                                       goto Exit
#define _AVI_MEM_REALLOC(Err)          if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Memory Re-allocation Error", AVI_MEM_REALLOC);}\
                                       Err = AVI_MEM_REALLOC;\
                                       goto Exit
#define _AVI_MEM_BUFFER_FULL(Err)      if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Memory Buffer Full", AVI_MEM_BUFFER_FULL);}\
                                       Err = AVI_MEM_BUFFER_FULL;\
                                       goto Exit
#define _AVI_OUT_FILE_OPEN(Err)        if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Output File Open Error", AVI_OUT_FILE_OPEN);}\
                                       Err = AVI_OUT_FILE_OPEN;\
                                       goto Exit
#define _AVI_HANDLE_IN_USE(Err)        if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Handle Already In Use", AVI_HANDLE_IN_USE);}\
                                       Err = AVI_HANDLE_IN_USE;\
                                       goto Exit
#define _AVI_FILE_NAME(Err)            if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Invalid File Name", AVI_FILE_NAME);}\
                                       Err = AVI_FILE_NAME;\
                                       goto Exit
#define _AVI_WRITE_FAIL(Err)           if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("File Write Error", AVI_WRITE_FAIL);}\
                                       Err = AVI_WRITE_FAIL;\
                                       goto Exit
#define _AVI_IN_FILE_OPEN(Err)         if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Input File Open Error", AVI_IN_FILE_OPEN);}\
                                       Err = AVI_IN_FILE_OPEN;\
                                       goto Exit
#define _AVI_MAX_OPEN_FILES(Err)       if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Number of Open Files Exceeded", AVI_MAX_OPEN_FILES);}\
                                       Err = AVI_MAX_OPEN_FILES;\
                                       goto Exit
#define _AVI_READ_FAIL(Err)            if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("File Read Error", AVI_READ_FAIL);}\
                                       Err = AVI_READ_FAIL;\
                                       goto Exit
#define _AVI_CORRUPT_FILE(Err)         if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Input File Corrupt", AVI_CORRUPT_FILE);} \
                                       Err = AVI_CORRUPT_FILE;\
                                       goto Exit
#define _AVI_FILE_RW(Err)              if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Read / Write Mismatch", AVI_FILE_RW);} \
                                       Err = AVI_FILE_RW;\
                                       goto Exit
#define _AVI_FILE_NOT_SUPPORTED(Err)   if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Unsupported Codec or File Type", AVI_FILE_NOT_SUPPORTED);}\
                                       Err = AVI_FILE_NOT_SUPPORTED;\
                                       goto Exit
#define _AVI_END_OF_SEQUENCE(Err)      if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Read Beyond End of Sequence", AVI_END_OF_SEQUENCE);}\
                                       Err = AVI_END_OF_SEQUENCE;\
                                       goto Exit
#define _AVI_DATA_ERROR(Err)           if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Image Data Error", AVI_DATA_ERROR);}\
                                       Err = AVI_DATA_ERROR;\
                                       goto Exit
#define _AVI_THREAD_ERROR(Err)         if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Multi-Threading Error", AVI_THREAD_ERROR);}\
                                       Err = AVI_THREAD_ERROR;\
                                       goto Exit
#define _AVI_THREAD_CS_ERROR(Err)      if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Thread Critical Section Error", AVI_THREAD_CS_ERROR);}\
                                       Err = AVI_THREAD_CS_ERROR;\
                                       goto Exit
#define _AVI_AUDIO_ERROR(Err)          if(NULL != gfnAVIErrorHandler){gfnAVIErrorHandler("Invalid audio parameters", AVI_AUDIO_ERROR);}\
                                       Err = AVI_AUDIO_ERROR;\
                                       goto Exit
/* Other macros */
#define _AVI_FreeAndNull( pointer )    if(NULL != pointer)\
                                       {  free( pointer );\
                                       }

/* Version 1.3.1, pHandleData->dwMoviChunkSize onwards added */
/* Version 1.4, pbFrameImageBuffer and pbFrameAudioBuffer initialised. pbInterlaceBuffer added and initialized */
#define _AVI_InitPrivateMembers        pHandleData->fFile = -1;\
                                       pHandleData->fVideo = 0;\
                                       pHandleData->fAudio = 0;\
                                       pHandleData->pASL_AudioIndexArray = NULL;\
                                       pHandleData->pASL_VideoIndexArray = NULL;\
                                       pHandleData->pVideoSuperIndexArray = NULL;\
                                       pHandleData->pVideoStandardIndexArray = NULL;\
                                       pHandleData->pAudioSuperIndexArray = NULL;\
                                       pHandleData->pAudioStandardIndexArray = NULL;\
                                       pHandleData->pszFileBufferName = NULL;\
                                       pHandleData->pbFrameBuffer = NULL;\
                                       pHandleData->pbAudioBuffer = NULL;\
                                       pHandleData->pbMemoryFileBuffer = NULL;\
                                       pHandleData->pbDataChunkHeaderBuffer = NULL;\
                                       pHandleData->qwMemoryFileSize = 0;\
                                       pHandleData->qwMemoryWritten = 0; \
                                       pHandleData->qfpBaseOffset = 484;\
                                       pHandleData->dwSuggestedBufferSize = 0;\
                                       pHandleData->dwAudioBufferSize = 0;\
                                       pHandleData->dwWidth = 0;\
                                       pHandleData->dwHeight = 0;\
                                       pHandleData->dwNumberOfSuperIndexes = 0;\
                                       pHandleData->dwFramesInChunk = 0;\
                                       pHandleData->dwNumberOfVideoChunks = 0;\
                                       pHandleData->dwNumberOfAudioChunks = 0;\
                                       pHandleData->dwInUseFlag = 0;\
                                       pHandleData->dwNewChunkFlag = 1;\
                                       pHandleData->dwAudioLength = 0;\
                                       pHandleData->dwChannels = 1;\
                                       pHandleData->fpStartOfHeader = (off64_t)(1670+32*_ASL_CHUNK_LIMIT);\
                                       pHandleData->dwMoviChunkSize = 0;\
                                       pHandleData->dwOffset = 0;\
                                       pHandleData->dwAudioBufferSize = 0;\
                                       pHandleData->qfpIndexOffset = 0;\
                                       pHandleData->dwFirstChunkSize = 0;\
                                       pHandleData->pbMemoryFramePointer = NULL;\
                                       pHandleData->pqwIndexPointer = NULL;\
                                       pHandleData->pbFrameImageBuffer = NULL;\
                                       pHandleData->pbFrameAudioBuffer = NULL;\
                                       pHandleData->pbInterlaceBuffer = NULL;\
                                       pHandleData->dwAudioSamplesPerRecord = 1;\
                                           

#define _AVI_InitPublicMembers         phFileHandle->pszFileBufferName = NULL;\
                                       phFileHandle->dwNumberOfFrames = 0;\
                                       phFileHandle->eRW = 0;\
                                       phFileHandle->eStoreType = 0;\
                                       phFileHandle->dwWidth = 0;\
                                       phFileHandle->dwHeight = 0;\
                                       phFileHandle->pbReadFrameBuffer = NULL;\
                                       phFileHandle->pbReadAudioBuffer = NULL;\
                                       phFileHandle->dwAudioBitsPerSample = 0;\
                                       phFileHandle->dwAudioSampleRate = 0;\
                                       phFileHandle->dwChannels = 0;\
                                       phFileHandle->dwMicroSecPerFrame = 0;\
                                       phFileHandle->dwMaxFrameSize = 0;\
                                       phFileHandle->pReserved = NULL;\

#define _AVI_RefreshPublicMembers      phFileHandle->pszFileBufferName = pHandleData->pszFileBufferName;\
                                       phFileHandle->dwNumberOfFrames = pHandleData->dwNumberOfFrames;\
                                       phFileHandle->eRW = pHandleData->eRW;\
                                       phFileHandle->eFileType = pHandleData->eFileType;\
                                       phFileHandle->eStoreType = pHandleData->eStoreType;\
                                       phFileHandle->dwWidth = pHandleData->dwWidth;\
                                       phFileHandle->dwHeight = pHandleData->dwHeight;\
                                       phFileHandle->pbReadFrameBuffer = pHandleData->pbFrameBuffer;\
                                       phFileHandle->pbReadAudioBuffer = pHandleData->pbAudioBuffer;\
                                       phFileHandle->dwAudioBitsPerSample = pHandleData->dwAudioBitsPerSample;\
                                       phFileHandle->dwAudioSampleRate = pHandleData->dwAudioSampleRate;\
                                       phFileHandle->dwChannels = pHandleData->dwChannels;\
                                       phFileHandle->dwMicroSecPerFrame = pHandleData->dwMicroSecPerFrame;\
                                       phFileHandle->dwMaxFrameSize = pHandleData->dwSuggestedBufferSize;\
                                     

/* Portable version of Microsoft Four Character Code (FOURCC) type and type conversion macro */
             
#define FOURCC ui32


/* Portable version of Microsoft BITMAPINFOHEADER. Uses unsigned types: none of these values will ever be negative */

typedef struct tagBitMapInfoHeader{
        ui32   dwSize;
        ui32   dwWidth;
        ui32   dwHeight;
        ui16   wPlanes;
        ui16   wBitCount;
        ui32   dwCompression;
        ui32   dwSizeImage;
        ui32   dwXPelsPerMeter;
        ui32   dwYPelsPerMeter;
        ui32   dwClrUsed;
        ui32   dwClrImportant;
        /*ui32   dwOffset;          Offset to JPEG specific fields. Optional parameter seems to be unused by any player*/
} BitMapInfoHeader;

/* Portable version of Microsoft PALETTEENTRY structure */

typedef struct tagPaletteEntry{
    ui8 bPeRed;
    ui8 bPeGreen;
    ui8 bPeBlue;
    ui8 bPeFlags;
} PaletteEntry;

/* Portable version of Microsoft AVIPALCHANGE palette change structure for use in video streams */
typedef struct tagAviPalChange{
    ui8		bFirstEntry;	      /* first entry to change */
    ui8		bNumEntries;	      /* # entries to change (0 if 256) */
    ui16		wFlags;		         /* Mostly to preserve alignment... */
    PaletteEntry bPeNew[256];	   /* New color specifications */
} AviPalChange;

/* Extended JPEG information structure. */

typedef struct tagJpegInfoHeader{   /* Optional data structure. Seems to be unused by any player */
   ui32 dwJpegSize;
   ui32 dwJpegProcess;
   ui32 dwJpegColorSpaceID;
   ui32 dwJpegHSubSampling;
   ui32 dwJpegVSubSampling;
} JpegInfoHeader;

/* Portable version of Microsoft WAVEFORMATEX header. */

typedef struct tagWaveFormatEx {
    ui16 wFormatTag;
    ui16 wChannels;
    ui32 dwSamplesPerSec;
    ui32 dwAvgBytesPerSec;
    ui16 wBlockAlign;
    ui16 wBitsPerSample;
    ui16 wExtendedSize;       /* Size, in bytes, of extra format information appended to the end of */
} WaveFormatEx;               /* the WAVEFORMATEX structure. Must be set to 0 for unformatted data. */

/* main header for the avi file (compatibility header)
 */
typedef struct tagAvi_Main_Header {
    FOURCC fcc;                     /* 'avih'                      */
    ui32  dwBlockSize;              /* size of this structure -8   */
    ui32  dwMicroSecPerFrame;       /* frame display rate (or 0L)  */
    ui32  dwMaxBytesPerSec;         /* max. transfer rate          */
    ui32  dwPaddingGranularity;     /* pad to multiples of this size; normally 2K */
    ui32  dwFlags;                  /* the ever-present flags      */
    #define AVIF_HASINDEX        0x00000010  /* Index at end of file? */
    #define AVIF_MUSTUSEINDEX    0x00000020
    #define AVIF_ISINTERLEAVED   0x00000100
    #define AVIF_TRUSTCKTYPE     0x00000800  /* Use CKType to find key frames */
    #define AVIF_WASCAPTUREFILE  0x00010000
    #define AVIF_COPYRIGHTED     0x00020000
    ui32  dwTotalFrames;            /* # frames in first movi list */
    ui32  dwInitialFrames;
    ui32  dwStreams;
    ui32  dwSuggestedBufferSize;
    ui32  dwWidth;
    ui32  dwHeight;
    ui32  dwReserved[4];
    } AVI_Main_Header;

typedef struct tagStream_Header {            /* 64 bytes */
   FOURCC fcc;          /* 'strh'  */
   ui32   dwBlockSize;  /* size of this structure - 8 */
   FOURCC fccType;      /* stream type codes */
   FOURCC fccHandler;
   ui32  dwFlags;

   #define AVISF_DISABLED          0x00000001
   #define AVISF_VIDEO_PALCHANGES  0x00010000

   ui16   wPriority;
   ui16   wLanguage;
   ui32  dwInitialFrames;
   ui32  dwScale;
   ui32  dwRate;       /* dwRate/dwScale is stream tick rate in ticks/sec */
   ui32  dwStart;
   ui32  dwLength;
   ui32  dwSuggestedBufferSize;
   ui32  dwQuality;
   ui32  dwSampleSize;
   struct {
      ui16 left;
      ui16 top;
      ui16 right;
      ui16 bottom;
      }   rcFrame;
   } Stream_Header;

typedef struct tagBaseIndexEntry{

   ui32 dwOffset;
   ui32 dwSize;

   }BaseIndexEntry;

typedef struct tagSuperIndex_Header{               /* 32 bytes */
   FOURCC   fcc;              /* 'indx' */
   ui32     dwBlockSize;      /* size of this structure - 8 */
   ui16     wLongsPerEntry;   /* ==4 */
   ui8      bIndexSubType;    /* ==0 (frame index) or AVI_INDEX_SUB_2FIELD */
   ui8      bIndexType;       /* ==AVI_INDEX_OF_INDEXES */
   ui32     dwEntriesInUse;   /* offset of next unused entry in aIndex */
   ui32     dwChunkId;        /* chunk ID of chunks being indexed */
   ui32     dwReserved[3];    /* must be 0 */
} SuperIndex_Header;

typedef struct tagSuperindex_Entry {               /* 16 bytes */
   off64_t  qfpOffset;     /* 64 bit offset to sub index chunk, i.e. to start of marker 'ix00' */
   ui32     dwSize;        /* 32 bit size of sub index chunk                                   */
   ui32     dwDuration;    /* time span of subindex chunk (in stream ticks). In this implementation this is the number of */
   } SuperIndex_Entry;     /* frames recorded within this index */

typedef struct tagStandardIndex_Header {           /* 32 bytes */
   FOURCC   fcc;                 /* 'ix##' */
   ui32     dwBlockSize;         /* size of this structure - 8 */
   ui16     wLongsPerEntry;      /* ==2 */
   ui8      bIndexSubType;       /* ==0 */
   ui8      bIndexType;          /* ==AVI_INDEX_OF_CHUNKS */
   ui32     dwEntriesInUse;      /* offset of next unused entry in aIndex */
   ui32     dwChunkId;           /* chunk ID of chunks being indexed, (e.g. 00db) */
   off64_t  qfpBaseOffset;       /* base offset that all index intries are relative to */
   ui32     dwReserved_3;        /* must be 0 */
   } StandardIndex_Header;

typedef struct tagStandardindex_Entry {            /* 8 bytes */
   ui32 dwOffset;                /* 32 bit offset to data (points to data, not riff header, e.g. to start of '00db' marker) */
   ui32 dwSize;                  /* 31 bit size of data (does not include size of riff header) */
   } StandardIndex_Entry;        /* Bit 31 is deltaframe bit, always 0 for MJPEG files */

typedef struct tagASL_Index_Entry {
   off64_t qfpOffset;            /* 64 bit Offset of start of data section from beginning of file */
   } ASL_Index_Entry;



typedef struct tagAviHandleData {
/* The following are "private" data members only accessible through the void *pReserved pointer in the file handle - if you 
 * know that this pointer points to an AviHandleData structure!
 */
   AviHandle *phHandle;
   fDesc fFile;                  /* Handle to opened file */
   ASL_Index_Entry *pASL_AudioIndexArray;
   ASL_Index_Entry *pASL_VideoIndexArray;
   SuperIndex_Entry *pVideoSuperIndexArray;
   StandardIndex_Entry *pVideoStandardIndexArray;
   SuperIndex_Entry *pAudioSuperIndexArray;
   StandardIndex_Entry *pAudioStandardIndexArray;
   ui32 dwInUseFlag;
   int  fVideo;                  /* Boolean flag defining whether there is video in the stream */
   int  fAudio;                  /* Boolean flag defining whether there is audio in the stream */
   ui32 dwNumberOfSuperIndexes;  /* Number of super indexes in use. Shouldn't have more than three (video, audio and text) */
   ui32 dwNumberOfVideoChunks;   /* Synonomous with the number of indexes in the super index structure for ASL generated files */
   ui32 dwNumberOfAudioChunks;
   ui32 dwFramesInChunk;         /* Number of frames recorded into the current chunk */
   ui32 dwNewChunkFlag;          /* Set if a new data chunk needs to be written. */
   ui32 dwSuggestedBufferSize;   /* Strictly, suggested buffer size for video */
   ui32 dwAudioBufferSize;       /* Suggested buffer size for audio - no longer fixed size from version 1.4. onwards to cope with third party files */
   ui32 dwMoviChunkSize;         /* Size of current movi chunk */
   ui32 dwOffset;                /* Offset in current movi chunk */
   ui32 dwAudioLength;           /* Length of entire (audio stream of) file in units of audio samples. */
   off64_t fpStartOfHeader;      /* Start of a chunks' header */
   off64_t qfpBaseOffset;        /* This is the offset to the start of the chunk index, as recorded in the super index */
   off64_t qfpIndexOffset;       /* This is the offset to the start of the first frame, as recorded in the chunk index */
   ui32 dwFirstChunkSize;
   ui8 *pbDataChunkHeaderBuffer; /* Points to memory allocated once only to buffer the chunk header writes */
   ui8 *pbFrameImageBuffer;      /* Single frame buffer used in "unbuffered" writes, i.e. writes where the entire video isn't buffered to memory */
   ui8 *pbFrameAudioBuffer;      /* Likewise for audio */
   ui8 *pbInterlaceBuffer;       /* Buffer used to merge two fields in an interlaced MJPEG file */
   ui32 dwAudioSamplesPerRecord; /* 1 for all files written using this library. Third party libraries may bundle lots of audio samples (frames) in one record */
   ui8 *pbMainHeaderBuffer;      /* Points to memory allocated to buffer main header reads */

   /* Used only when writing to memory */

   ui8  *pbMemoryFileBuffer;     /* Pointer to malloced area used to write / read successive frame information.  */
   ui8  *pbMemoryFramePointer;   /* Points to actual frame data in the malloced area */
   ui64 *pqwIndexPointer;        /* Points to index at the start of the malloced area */
   ui64 qwMemoryFileSize;        /* Size of buffer to hold buffered write */
   ui64 qwMemoryWritten;         /* Amount of data actually written to the memory file */


/* The following are duplicates of the public data members, held so that the user cannot (easily) modify important
   parameters and cause problems.
 */
   char  *pszFileBufferName;  /* Pointer to file / buffer name */
   eReadWrite eRW;            /* Flag to show whether the file is read or write */
   eFile_Type eFileType;      /* File type */
   ui32 dwNumberOfFrames;     /* Total number of frames in the file */
   ui32 dwWidth;              /* Frame width  */
   ui32 dwHeight;             /* Frame height */
   ui32 dwMicroSecPerFrame;   /* Proxy for file frame rate */
   ui32 dwChannels;           /* 1 for mono, 2 for stereo. Defaults to 1.*/
   ui32 dwAudioSampleRate;    /* Samples per second */
   ui32 dwAudioBitsPerSample; /* Bits per sample, should be 8 or 16 */
   ui8  *pbFrameBuffer;       /* Pointer to frame buffer used with this file */
   ui8  *pbAudioBuffer;       /* Pointer to audio buffer used with this file. */
   eStore_Type eStoreType;    /* Specifies whether the file is in volatile memory or on disk */
   } AviHandleData;

AVI_ERR AVI_WriteHeader(AviHandleData *pHandleData);
AVI_ERR WriteDataChunkHeader(AviHandleData *pHandleData);
AVI_ERR WriteDataChunk(AviHandleData *pHandleData);


AVI_ERR AVI_READ2(fDesc fFile, ui16 *wTarget);
/* Read data from file byte at a time and convert to ui16.
 * Independent of processor endiannes
 */

AVI_ERR AVI_READ4(fDesc fFile, ui32 *dwTarget);
/* Read data from file byte at a time and convert to ui32.
 * Independent of processor endiannes
 */

AVI_ERR AVI_READ8(fDesc fFile, off64_t *qfpTarget);
/* Read data from file byte at a time and convert to ui64.
 * Independent of processor endiannes
 */


ui8 * AVI_READ2_BUFFER(ui8 *pInputBuffer, ui16 *wTarget);
/* Read data from buffer byte at a time and convert to ui16.
 * Independent of processor endiannes
 */

ui8 * AVI_READ4_BUFFER(ui8 *pInputBuffer, ui32 *dwTarget);
/* Read data from buffer byte at a time and convert to ui32.
 * Independent of processor endiannes
 */

ui8 * AVI_READ8_BUFFER(ui8 *pInputBuffer, off64_t *qfpTarget);
/* Read data from buffer byte at a time and convert to ui64.
 * Independent of processor endiannes
 */

ui8 * AVI_WRITE2_BUFFER(ui16 *wSource, ui8 *pBuffer);
/* Write data to buffer byte at a time and convert to ui16.
 * Independent of processor endiannes.
 * New buffer position returned.
 */

ui8 * AVI_WRITE4_BUFFER(ui32 *dwSource, ui8 *pBuffer);
/* Write data to buffer byte at a time and convert to ui32.
 * Independent of processor endiannes.
 * New buffer position returned.
 */

ui8 * AVI_WRITE8_BUFFER(off64_t *qfpSource, ui8 *pBuffer);
/* Write data to buffer byte at a time and convert to ui64.
 * Independent of processor endiannes.
 * New buffer position returned.
 */

ui8 * AVI_MEMCOPY(ui8 *pbDestBuffer, ui8 *pbSourceBuffer, ui32 dwCopySize);
/* Copies data between buffers, also incrementing destination buffer pointer.
 */

FOURCC FCC(char *pszFourCharacter);


/* These frame write functions are kept private. The library works out which one to call based on whether the file has been opened
 * for buffered output or write to disk.
 */

AVI_ERR AVI_BufferedWriteNextFrame(AviHandle *phFileHandle, ui8 *pbBuffer, ui32 dwFrameSize, ui8 *pbAudioBuffer);

AVI_ERR AVI_UnBufferedWriteNextFrame(AviHandle *phFileHandle, ui8 *pbBuffer, ui32 dwFrameSize, ui8 *pbAudioBuffer);

/* Private functions to parse the video and audio super indexes.
 */

AVI_ERR AVI_ParseAudioSuperIndex(AviHandle *phFileHandle, fDesc iFile, ui8 *pbAudioBuffer, AVI_Main_Header sAviMainHeader);
AVI_ERR AVI_ParseVideoSuperIndex(AviHandle *phFileHandle, fDesc iFile, AVI_Main_Header sAviMainHeader, Stream_Header sStreamHeader);

/* Private functions to check for the presence of a valid "ix", "db", "dc" and "wb" markers
 */
AVI_ERR AVI_CheckForIX_Marker(FOURCC fcc);
AVI_ERR AVI_CheckForWB_Marker(FOURCC fcc);
AVI_ERR AVI_CheckForVid_Marker(FOURCC fcc);

