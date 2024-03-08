/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : pbl_api.h
 * Function    : User API for Phoenix Buffer Library
 * Project     : AP06
 * Authors     : Richard Brown, Martin Bone
 * Systems     : ANSI C, Win32 (MSVC), 32 bit DOS (Symantec).
 * Version     : 1.0
 * Release Date: 
 *
 * Copyright (c) 2004 Active Silicon Ltd.
 ****************************************************************************
 * Comments:
 * --------
 */
#ifndef _PBL_API_H_
#define _PBL_API_H_

/* Include all the OS dependent definitions */
/* Need to use #import if using native MacOS X with frameworks */
#if defined _ASL_MACOSX_FMWK
#import <ASLphoenix/phx_os.h>

#else
#include <phx_os.h>

#endif


/* These are the parameters that can be obtained from a
 * PBL buffer.
 */
typedef enum {
   PBL_BUFF_WIDTH       = (int)1,
   PBL_BUFF_HEIGHT      = (int)2,
   PBL_BUFF_ADDRESS     = (int)3,
   PBL_BUFF_STRIDE      = (int)4,
   PBL_PALETTE          = (int)5,
   PBL_WINDOW           = (int)6,
   PBL_DST_FORMAT       = (int)7,
   PBL_BUFF_FORMAT      = (int)8,
   PBL_BUFF_BAYDEC      = (int)9,
   PBL_BUFF_BAYCOL      = (int)10,
   PBL_BUFF_POINTER     = (int)11,
   PBL_BUFF_DISPLAY     = (int)12,
   PBL_BUFF_EMODE       = (int)13,
   PBL_BUFF_SURFACE     = (int)14,
   PBL_BUFF_GRCTX       = (int)15,
   PBL_PIXEL_FORMAT     = (int)16,
   PBL_PS_DD            = (int)17
} etPblParam;   

/* These are the parameter values that can be obtained from a
 * PBL buffer.
 */
typedef enum {
   PBL_BAY_DEC_DUP   =  (int)0x100 + PBL_BUFF_BAYDEC,
   PBL_BAY_DEC_AVE   =  (int)0x200 + PBL_BUFF_BAYDEC,
   PBL_BAY_DEC_MED   =  (int)0x300 + PBL_BUFF_BAYDEC,
   PBL_BAY_COL_RED   =  (int)0x100 + PBL_BUFF_BAYCOL,
   PBL_BAY_COL_GRNR  =  (int)0x200 + PBL_BUFF_BAYCOL,
   PBL_BAY_COL_GRNB  =  (int)0x300 + PBL_BUFF_BAYCOL,
   PBL_BAY_COL_BLU   =  (int)0x400 + PBL_BUFF_BAYCOL
} etPblParamValue;   

/* PBL buffers can be created either in system memory or in
 * the memory on the video card. Additionally, Phoenix can be instructed
 * to capture directly into the display buffer, or to an external
 * buffer. This second case would be used when the user wants to
 * capture to his/her own buffer, perform some image processing
 * and then copy the data to the display buffer.
 */
typedef enum {
   PBL_BUFF_VIDCARD_MEM_DIRECT,     /* Video card memory, direct capture */
   PBL_BUFF_VIDCARD_MEM_INDIRECT,   /* Video card memory, indirect capture */
   PBL_BUFF_SYSTEM_MEM_DIRECT,      /* System memory, direct capture */
   PBL_BUFF_SYSTEM_MEM_INDIRECT     /* System memory, indirect capture */
} etPblBufferMode;



/* Prototype Definitions
 * =====================
 */

#if defined __cplusplus
extern "C" {
#endif

etStat PHX_EXPORT_FN PBL_BufferCreate ( tPHX*, etPblBufferMode, tPHX, tHandle, void (*) (const char*, etStat, const char*) );
etStat PHX_EXPORT_FN PBL_BufferDestroy( tPHX* );
etStat PHX_EXPORT_FN PBL_BufferInit   ( tPHX  );
etStat PHX_EXPORT_FN PBL_BufferParameterSet( tPHX, etPblParam, void* );
etStat PHX_EXPORT_FN PBL_BufferParameterGet( tPHX, etPblParam, void* );

#if defined __cplusplus
};
#endif

#endif  /* _PBL_API_H_ */

