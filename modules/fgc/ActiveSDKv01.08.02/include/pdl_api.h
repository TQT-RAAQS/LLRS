/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : pdl_api.h
 * Function    : User API for Phoenix display library
 * Project     : AP06
 * Authors     : Warren Frost, Martin Bone
 * Systems     : ANSI C, Win32 (MSVC), 32 bit DOS (Symantec).
 * Version     : 1.0
 * Release Date: 14-Feb-2002
 *
 * Copyright (c) 2002-2004 Active Silicon Ltd.
 ****************************************************************************
 * Comments:
 * --------
 */
#ifndef _PDL_API_H_
#define _PDL_API_H_

/* Include all the OS dependent definitions */
/* Need to use #import if using native MacOS X with frameworks */
#if defined _ASL_MACOSX_FMWK
#import <ASLphoenix/phx_os.h>

#else
#include <phx_os.h>

#endif


/* These are the parameters that can be obtained from a
 * PDL display or buffer.
 */
typedef enum {
   PDL_BUFF_WIDTH       = (int) 1,
   PDL_BUFF_HEIGHT      = (int) 2,
   PDL_BUFF_ADDRESS     = (int) 3,
   PDL_BUFF_STRIDE      = (int) 4,
   PDL_PALETTE          = (int) 5,
   PDL_WINDOW           = (int) 6,
   PDL_DST_FORMAT       = (int) 7,
   PDL_BUFF_LAST        = (int) 8,
   PDL_PIXEL_FORMAT     = (int) 9,
   PDL_PS_DD            = (int)10,
   PDL_F_QUIT           = (int)11,
   PDL_X_POS            = (int)12,
   PDL_Y_POS            = (int)13,
   PDL_SFC_PRIMARY      = (int)14,
   PDL_X_OFFSET         = (int)15,
   PDL_Y_OFFSET         = (int)16,
   PDL_BYTES_PER_PIXEL  = (int)17,
   PDL_XWIN_DISPLAY     = (int)18
} etPdlParam;   

/* Display buffers can be created either in system memory or in
 * the memory on the video card. Additionally, Phoenix can be instructed
 * to capture directly into the display buffer, or to an external
 * buffer. This second case would be used when the user wants to
 * capture to his/her own buffer, perform some image processing
 * and then copy the data to the display buffer.
 */
typedef enum {
   PDL_BUFF_VIDCARD_MEM_DIRECT,     /* Video card memory, direct capture */
   PDL_BUFF_VIDCARD_MEM_INDIRECT,   /* Video card memory, indirect capture */
   PDL_BUFF_SYSTEM_MEM_DIRECT,      /* System memory, direct capture */
   PDL_BUFF_SYSTEM_MEM_INDIRECT     /* System memory, indirect capture */
} etBufferMode;



/* Prototype Definitions
 * =====================
 */

#if defined __cplusplus
extern "C" {
#endif

etStat PHX_EXPORT_FN PDL_DisplayCreate ( tPHX*, void *, tHandle, void (*) (const char*, etStat, const char*) );
etStat PHX_EXPORT_FN PDL_DisplayDestroy( tPHX* );
etStat PHX_EXPORT_FN PDL_DisplayInit( tPHX );
etStat PHX_EXPORT_FN PDL_DisplayParameterSet( tPHX, etPdlParam, void* );
etStat PHX_EXPORT_FN PDL_DisplayParameterGet( tPHX, etPdlParam, void* );
etStat PHX_EXPORT_FN PDL_DisplayPaletteSet( tPHX );


etStat PHX_EXPORT_FN PDL_BufferCreate( tPHX*, tPHX, etBufferMode );
etStat PHX_EXPORT_FN PDL_BufferDestroy( tPHX* );
etStat PHX_EXPORT_FN PDL_BufferPaint( tPHX );
etStat PHX_EXPORT_FN PDL_BufferParameterSet( tPHX, etPdlParam, void* );
etStat PHX_EXPORT_FN PDL_BufferParameterGet( tPHX, etPdlParam, void* );

#if defined __cplusplus
};
#endif

#endif  /* _PDL_API_H_ */

