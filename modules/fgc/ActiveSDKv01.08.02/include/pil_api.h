/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : pil_api.h
 * Function    : User API for Phoenix image processing library
 * Project     : AP06
 * Authors     : Richard Brown, Martin Bone
 * Systems     : ANSI C, Win32 (MSVC), 32 bit DOS (Symantec).
 * Version     : 1.0
 * Release Date:
 *
 * Copyright (c) 2004-2008 Active Silicon Ltd.
 ****************************************************************************
 * Comments:
 * --------
 */
#ifndef _PIL_API_H_
#define _PIL_API_H_

/* Include all the OS dependent definitions */
/* Need to use #import if using native MacOS X with frameworks */
#if defined _ASL_MACOSX_FMWK
#import <ASLphoenix/phx_os.h>

#else
#include <phx_os.h>

#endif

#define stPilPalette HPALETTE

typedef enum {
    PIL_FILE_BMP,
    PIL_FILE_JPEG,
    PIL_FILE_RAW,
    PIL_FILE_TIFF
} etPilFileFormat;

typedef enum {
    PIL_LINE_DUPLICATION,
    PIL_LINE_AVERAGE,
    PIL_BILINEAR_INTERPOLATION,
    PIL_MEDIAN_FILTER
} etPilInterpolation;

/* Prototype Definitions
 * =====================
 */

#if defined __cplusplus
extern "C" {
#endif

etStat PHX_EXPORT_FN PIL_Convert(tPHX, tPHX);
etStat PHX_EXPORT_FN PIL_FieldToFrame(tPHX, tPHX, etPilInterpolation,
                                      tFlag fFirstField);
etStat PHX_EXPORT_FN PIL_FileWrite(tPHX, char *pszFileName,
                                   etPilFileFormat eFileFormat);

#if defined __cplusplus
};
#endif

#endif /* _PIL_API_H_ */
