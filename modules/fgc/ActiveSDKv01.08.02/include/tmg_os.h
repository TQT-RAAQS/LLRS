/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : tmg_os.h
 * Function    : All definitions that vary with operating system environment.
 * Project     : General
 * Authors     : Colin Pearce
 * Systems     : ANSI C, Various.
 * Version     : See tmg_api.h for overall library release information.
 * Release Date: See tmg_api.h for overall library release information.
 *
 * Copyright (c) 1993-99 Active Silicon Ltd.
 ****************************************************************************
 *
 * File History
 * ------------
 * 12-Jul-99, HCP, Created during TMG include file restructuring.
 *
 *
 * Comments:
 * --------
 * This file contains sections of code for each of the operating system
 * environments that it supports (e.g. Win32).
 *
 * Supported operating environments are:
 * ------------------------------------
 * 1. "_TMG_DOS32"   - 32 bit DOS (Watcom & Symantec).
 * 2. "_TMG_WINDOWS" - Windows 32 and 64 bit.
 * 3. "_TMG_LINUX"   - Linux 32 and 64 bit.
 * 4. "_TMG_MACOSX"  - Mac OS X, 32 and 64 bit.
 * 5. "_TMG_QNX4"    - QNX 4.
 * 6. "_TMG_QNX6"    - QNX 6.
 * 7. "_TMG_ARM7"    - ARM7 processors (and Arm SDT).
 *
 *
 * The sections within each operating environment are:
 * 1. Include files.
 * 2. Malloc and free macros.
 * 3. General type definitions -"ui32" etc definitions.
 * 4. Library export and calling convention definitions.
 * 5. Error print macros - for pop-up window / printf etc.
 * 6. Assertion macros (included if _TMG_DEBUG defined).
 * 7. Debug macros.
 *
 ****************************************************************************
 */
#ifndef _TMG_OS_H_
#define _TMG_OS_H_

#define TMG_DEBUG_STRING_LEN 128

#if defined _TMG_DOS32 /*=================================================*/

/*
_TMG_DOS32(ForBrief)
-------------------
*/

/* Special DOS32 related definitions:
 */
#ifdef __WATCOMC__
#define _TMG_FG_GRAPHICS /* Watcom command length problems */
#endif

/* 1. Include Files
 * ----------------
 */
#include <conio.h>
#include <io.h>
#if defined __WATCOMC__
#include <stddef.h> /* for cdecl calling convention */
#endif
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <malloc.h>
#include <math.h>
#include <memory.h>
#include <stdarg.h> /* Must be before stdio.h for MS Windows */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Flashtek Stuff
 * --------------
 */
#if defined _TMG_FG_GRAPHICS
#include <fg.h> /* Always included for DOS32 - to keep structure same */
#endif
#ifndef __GNUC__
#include <x32.h> /* 32 bit DOS may use the X-32VM extender */
#endif

/* GRX graphics */
#if defined _TMG_GRX_GRAPHICS
#include <grx20.h>
#endif

/* 2. Malloc/free macros
 * ---------------------
 */
#ifdef _TMG_DEBUG
#define _TMG_Malloc(x) TMG_DebugMalloc(x)
#define _TMG_Free(x) TMG_DebugFree(x)
#else
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)
#endif

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
typedef long i32;
typedef unsigned long ui32;
typedef long m16;
typedef unsigned long mu16;
typedef long m32;
typedef unsigned long mu32;
#define _ASL_TYPEDEFS
#endif

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#ifdef __WATCOMC__ /* stack based calling convention only supported */
#define EXPORT_FN __cdecl
#define EXPORT_FN_PTR __cdecl *
#else
#define EXPORT_FN
#define EXPORT_FN_PTR *
#endif /* not Watcom */

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { printf("ERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _TMG_Assert assert

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)                     \
    {                                                                          \
        printf("TMG_Debug (%s) - %s: 0x%02lx (%02ld) : 0x%02lx (%02ld)\n",     \
               szFnName, pszMessage, (ui32)dwParameter1, (ui32)dwParameter1,   \
               (ui32)dwParameter2, (ui32)dwParameter2);                        \
    }

/* _TMG_DebugString: Outputs a string as part of a debug string.
   ---------------- */
#define _TMG_DebugString(pszMessage, pszParameter)                             \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        wsprintf(_szParamString, " %s\r", (ui32)pszParameter);                 \
        strcat(_szWork, _szParamString);                                       \
        OutputDebugString(_szWork);                                            \
    }

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)                                  \
    { ::MessageBox(NULL, pszMessage, pszTitle, MB_OK); }
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)                                  \
    { MessageBox(NULL, pszMessage, pszTitle, MB_OK); }
#endif

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    (void)/*lint -restore */

#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    (void)/*lint -restore */

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#elif defined _TMG_WINDOWS /*=================================================*/

/*
_TMG_WINDOWS(ForBrief)
-------------------
*/
#define _CRT_SECURE_NO_DEPRECATE /*999: Ignore deprecated warnings in VS2008   \
                                  */

/* 1. Include Files
 * ----------------
 */
#include <windows.h>             /* windows.h must be included first */
#ifdef _WINDOWS                  /* This is an MS Windows definition -    */
#include <ddraw.h>               /* For Windows NT (4) Direct Draw API    */
#include <windowsx.h>            /* not defined for console mode programs */
#endif                           /* _WINDOWS */

/*#include <conio.h> NOT ANSI */
#include <ctype.h>
#include <errno.h>
#include <io.h>
#include <malloc.h>
#include <math.h>
#include <stdarg.h> /* must be before stdio.h for MS Windows */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/*#include <memory.h> NOT ANSI */
#include <assert.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#ifdef _TMG_DEBUG
#define _TMG_Malloc(x) TMG_DebugMalloc(x)
#define _TMG_Free(x) TMG_DebugFree(x)
#else
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)
#endif

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
typedef long i32;
typedef unsigned long ui32;
typedef long m16;
typedef unsigned long mu16;
typedef long m32;
typedef unsigned long mu32;
typedef signed _int64 i64;
typedef unsigned _int64 ui64;
typedef _int64 off64_t;
typedef int fDesc;
#define _ASL_TYPEDEFS
#endif

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define EXPORT_FN __declspec(dllexport)
#define EXPORT_FN_PTR __declspec(dllexport) *

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#ifdef __cplusplus
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { ::MessageBox(NULL, szMessage, szTitle, MB_OK); }
#else
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { MessageBox(NULL, szMessage, szTitle, MB_OK); }
#endif

/* 6. Assert Macros
 * ----------------
 * The Microsoft ANSI C "assert" is not compiled in when NDEBUG is selected,
 * which is selected by default on the project makefiles for release builds.
 */
#define _TMG_Assert assert

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    (void)/*lint -restore */
/* #define _TMG_DPRINTF(L)  if (L>0) printf */

#ifdef _CONSOLE /* Win32 automatic definition */

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)                     \
    {                                                                          \
        printf("TMG_Debug (%s) - %s: 0x%02lx (%02ld) : 0x%02lx (%02ld)\n",     \
               szFnName, pszMessage, (ui32)dwParameter1, (ui32)dwParameter1,   \
               (ui32)dwParameter2, (ui32)dwParameter2);                        \
    }

/* _TMG_DebugString: Outputs a string as part of a debug string.
   ---------------- */
#define _TMG_DebugString(pszMessage, pszParameter)                             \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        wsprintf(_szParamString, " %s\r", (ui32)pszParameter);                 \
        strcat(_szWork, _szParamString);                                       \
        OutputDebugString(_szWork);                                            \
    }

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)                                  \
    { ::MessageBox(NULL, pszMessage, pszTitle, MB_OK); }
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)                                  \
    { MessageBox(NULL, pszMessage, pszTitle, MB_OK); }
#endif

#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#else /* #else part of "#ifdef _CONSOLE"   */

#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)                     \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        (void)dwParameter2;                                                    \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        wsprintf(_szParamString, " 0x%lx (%ld)\n", (ui32)dwParameter1,         \
                 (ui32)dwParameter1);                                          \
        strcat(_szWork, _szParamString);                                       \
        OutputDebugString(_szWork);                                            \
    }

/* _TMG_DebugString: Outputs a string as part of a debug string.
   ---------------- */
#define _TMG_DebugString(pszMessage, pszParameter)                             \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        wsprintf(_szParamString, " %s\r", (ui32)pszParameter);                 \
        strcat(_szWork, _szParamString);                                       \
        OutputDebugString(_szWork);                                            \
    }

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   --------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)                                  \
    { ::MessageBox(NULL, pszMessage, pszTitle, MB_OK); }
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)                                  \
    { MessageBox(NULL, pszMessage, pszTitle, MB_OK); }
#endif

#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#endif /* #else part of "#ifdef _CONSOLE"  */

#elif defined _TMG_LINUX /*=================================================*/

/*
_TMG_LINUX(ForBrief)
--------------------
*/

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/file.h>
#include <sys/mman.h>
#include <sys/time.h> /* struct timeval & gettimeofday */
#include <sys/types.h>

#ifdef _TMG_LINUX_XSHM
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

/* X Imaging under Linux
 * -----------------------
 */
#if defined _TMG_X_GRAPHICS
#include <X11/X.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xos.h>
#include <X11/Xutil.h>
#if defined _TMG_GL_GRAPHICS
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glx.h>
#endif
#endif

#ifdef _TMG_LINUX_XSHM
#include <X11/extensions/XShm.h>
#include <X11/extensions/Xext.h>
#endif

/* 2. Malloc/free macros
 * ---------------------
 */
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
/* Note: int is 32 bit (as is long) but long generates a compiler warning */
typedef int i32;
typedef unsigned int ui32;
typedef long m16;
typedef unsigned long mu16;
typedef long m32;
typedef unsigned long mu32;
typedef long long i64;
typedef unsigned long long ui64;
/* typedef long long   off64_t; [defined elsewhere in Linux, Colin, Mar-12] */
typedef int fDesc;
#define _ASL_TYPEDEFS
#endif

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define EXPORT_FN
#define EXPORT_FN_PTR *

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _TMG_Assert assert

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)                     \
    {                                                                          \
        printf("TMG_Debug (%s) - %s: 0x%02lx (%02ld) : 0x%02lx (%02ld)\n",     \
               szFnName, pszMessage, (ui32)dwParameter1, (ui32)dwParameter1,   \
               (ui32)dwParameter2, (ui32)dwParameter2);                        \
    }

/* _TMG_DebugString: Outputs a string as part of a debug string.
   --------------- */
#define _TMG_DebugString(pszMessage, pszParameter)                             \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        sprintf(_szParamString, " %s\r", (ui32)pszParameter);                  \
        strcat(_szWork, _szParamString);                                       \
        printf(_szWork);                                                       \
    }

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)
#endif

#define _TMG_DPRINTF(L)                                                        \
    if (L > 10)                                                                \
    printf

#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    (void)/*lint -restore */

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#elif defined _TMG_MACOSX /*=================================================*/
/*
_TMG_MACOSX(ForBrief)
-------------------
*/

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined __MWERKS__
#import <Carbon/Carbon.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>
#import <OpenGL/glext.h>
#import <OpenGL/glu.h>
#endif
/* 2. Malloc/free macros
 * ---------------------
 */
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS

typedef unsigned char ui8;
typedef signed char i8;
typedef unsigned short ui16;
typedef signed short i16;
#if __LP64__
typedef unsigned int ui32;
typedef signed int i32;
#else
typedef unsigned long ui32;
typedef signed long i32;
#endif

typedef unsigned long long ui64;
typedef signed long long i64;
typedef i16 m16;
typedef ui16 mu16;
typedef i32 m32;
typedef ui32 mu32;

#if defined __LP64__
typedef ui64 ptr_t;
#else
typedef ui32 ptr_t;
#endif

#define _ASL_TYPEDEFS
#endif

typedef off_t
    off64_t; /* This seems to be a signed 64 bit integer in the MAC world */
typedef int fDesc;

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define EXPORT_FN
#define EXPORT_FN_PTR *

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 *
 * Mac alerts require special string object handlling, therefore a
 * separate function is used in tmg_macx.c
 */
#if !defined __MWERKS__ && !defined(__LP64__)
extern void TMG_DRV_ErrPrint(char *, char *);
#define _TMG_ErrPrint(szMessage, szTitle) TMG_DRV_ErrPrint(szMessage, szTitle)
#else
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    {}
#endif

/* 6. Assert Macros
 * ----------------
 */
#define _TMG_Assert assert

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)

/* _TMG_DebugString: Outputs a string as part of a debug string.
   ---------------- */
#define _TMG_DebugString(pszMessage, pszParameter)

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)
#endif

#define _TMG_DPRINTF(L)                                                        \
    if (L > 10)                                                                \
    printf
#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    printf /*lint -restore */

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#elif defined _TMG_VXWORKS /*=================================================*/

/*
_TMG_VXWORKS(ForBrief)
--------------------
*/

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
/* Note: int is 32 bit (as is long) but long generates a compiler warning */
typedef int i32;
typedef unsigned int ui32;
typedef long m16;
typedef unsigned int mu16;
typedef long m32;
typedef unsigned int mu32;
#define _ASL_TYPEDEFS
#endif

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define EXPORT_FN
#define EXPORT_FN_PTR *

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _TMG_Assert assert

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)                     \
    {                                                                          \
        printf("TMG_Debug (%s) - %s: 0x%02lx (%02ld) : 0x%02lx (%02ld)\n",     \
               szFnName, pszMessage, (ui32)dwParameter1, (ui32)dwParameter1,   \
               (ui32)dwParameter2, (ui32)dwParameter2);                        \
    }

/* _TMG_DebugString: Outputs a string as part of a debug string.
   --------------- */
#define _TMG_DebugString(pszMessage, pszParameter)                             \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        sprintf(_szParamString, " %s\r", (ui32)pszParameter);                  \
        strcat(_szWork, _szParamString);                                       \
        printf(_szWork);                                                       \
    }

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)
#endif

#define _TMG_DPRINTF(L)                                                        \
    if (L > 10)                                                                \
    printf

#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#define _TMG_DPRINTF(L)                                                        \
    if (FALSE)                                                                 \
    printf

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#elif defined _TMG_QNX4 /*=================================================*/

/*
_TMG_QNX4(ForBrief)
--------------------
*/

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
/* Note: int is 32 bit (as is long) but long generates a compiler warning */
typedef int i32;
typedef unsigned long ui32;
typedef long m16;
typedef unsigned long mu16;
typedef long m32;
typedef unsigned long mu32;
#define _ASL_TYPEDEFS
#endif

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define EXPORT_FN
#define EXPORT_FN_PTR *

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _TMG_Assert assert

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)                     \
    {                                                                          \
        printf("TMG_Debug (%s) - %s: 0x%02lx (%02ld) : 0x%02lx (%02ld)\n",     \
               szFnName, pszMessage, (ui32)dwParameter1, (ui32)dwParameter1,   \
               (ui32)dwParameter2, (ui32)dwParameter2);                        \
    }

/* _TMG_DebugString: Outputs a string as part of a debug string.
   --------------- */
#define _TMG_DebugString(pszMessage, pszParameter)                             \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        sprintf(_szParamString, " %s\r", (ui32)pszParameter);                  \
        strcat(_szWork, _szParamString);                                       \
        printf(_szWork);                                                       \
    }

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)
#endif

#define _TMG_DPRINTF(L)                                                        \
    if (L > 10)                                                                \
    printf

#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    (void)/*lint -restore */

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#elif defined _TMG_QNX6 /*=================================================*/

/*
_TMG_QNX6(ForBrief)
--------------------
*/

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
/* Note: int is 32 bit (as is long) but long generates a compiler warning */
typedef int i32;
typedef unsigned int ui32;
typedef int m16;
typedef unsigned int mu16;
typedef int m32;
typedef unsigned int mu32;
#define _ASL_TYPEDEFS
#endif

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define EXPORT_FN
#define EXPORT_FN_PTR *

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _TMG_Assert assert

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)                     \
    {                                                                          \
        printf("TMG_Debug (%s) - %s: 0x%02lx (%02ld) : 0x%02lx (%02ld)\n",     \
               szFnName, pszMessage, (ui32)dwParameter1, (ui32)dwParameter1,   \
               (ui32)dwParameter2, (ui32)dwParameter2);                        \
    }

/* _TMG_DebugString: Outputs a string as part of a debug string.
   --------------- */
#define _TMG_DebugString(pszMessage, pszParameter)                             \
    {                                                                          \
        char _szWork[TMG_DEBUG_STRING_LEN];                                    \
        char _szParamString[TMG_DEBUG_STRING_LEN];                             \
        strcpy(_szWork, szFnName);                                             \
        strcat(_szWork, " - ");                                                \
        strcat(_szWork, pszMessage);                                           \
        sprintf(_szParamString, " %s\r", (ui32)pszParameter);                  \
        strcat(_szWork, _szParamString);                                       \
        printf(_szWork);                                                       \
    }

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)
#endif

#define _TMG_DPRINTF(L)                                                        \
    if (L > 10)                                                                \
    printf

#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    (void)/*lint -restore */

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#elif defined _TMG_ARM7 /*=================================================*/

/*
_TMG_ARM7(ForBrief)
-------------------
*/

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#ifdef _TMG_DEBUG
#define _TMG_Malloc(x) TMG_DebugMalloc(x)
#define _TMG_Free(x) TMG_DebugFree(x)
#else
#define _TMG_Malloc(x) malloc(x)
#define _TMG_Free(x) free(x)
#endif

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
/* Note: int is 32 bit (as is long) but long generates a compiler warning */
typedef int i32;
typedef unsigned int ui32;
typedef long m16;
typedef unsigned long mu16;
typedef long m32;
typedef unsigned long mu32;
#define _ASL_TYPEDEFS
#endif

typedef ui32 Terr;
typedef ui32 Thandle;
typedef ui32 Tboolean;
typedef ui32 Tparam;
typedef ui32 Timage_handle;
typedef ui32 Tdisplay_handle;

#define IM_UI8 ui8
#define IM_UI16 ui16
#define IM_UI32 ui32
#define IM_I32 i32
#define IM_I16 i16
#define CMAP_PTR struct Tcmap

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define EXPORT_FN       /* __cdecl  */
#define EXPORT_FN_PTR * /* __cdecl* */

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _TMG_ErrPrint(szMessage, szTitle)                                      \
    { printf("ERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _TMG_Assert assert

/* 7. Debug Macros
 * ---------------
 */
#ifdef _TMG_DEBUG

/* _TMG_Debug: Outputs a long word as part of a debug string.
   ---------- */
#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)

/* _TMG_DebugString: Outputs a string as part of a debug string.
   ---------------- */
#define _TMG_DebugString(pszMessage, pszParameter)

/* _TMG_DebugPopup: Outputs a debug string in a popup.
   -------------- */
#ifdef __cplusplus
#define _TMG_DebugPopup(pszMessage, pszTitle)
#else
#define _TMG_DebugPopup(pszMessage, pszTitle)
#endif

#define _TMG_DPRINTF(L)                                                        \
    if (L > 10)                                                                \
    printf
#else /* no debugging - _TMG_DEBUG not defined */

#define _TMG_Debug(pszMessage, dwParameter1, dwParameter2)
#define _TMG_DebugString(pszMessage, pszParameter)
#define _TMG_DebugPopup(pszMessage, pszTitle)

#define _TMG_DPRINTF(L) /*lint -save -e505 -e774 */                            \
    if (FALSE)                                                                 \
    (void)/*lint -restore */

#endif /* #else part of "#ifdef _TMG_DEBUG" */

#else

#error You have a missing target environment directive (e.g. _TMG_WINDOWS)

#endif

/* Windows 3.1 "Huge" Pointer Compatibility
 * ----------------------------------------
 * If these haven't been specially defined by Windows 3.1 OS option, then
 * define them as standard types.
 * (For windows 3.1, IM_UI8 would "normally" have the "huge" modifier.)
 * Note: We'll keep this for now - it may be useful sometime.
 */
#ifndef IM_UI8
#define IM_UI8 ui8
#define IM_UI16 ui16
#endif

#endif /* _TMG_OS_H_ */
