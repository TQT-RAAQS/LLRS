/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : phx_os.h
 * Function    : All definitions that vary with operating system environment.
 * Project     : AP06
 * Authors     : Martin Bone, Richard Brown
 * Systems     : ANSI C
 * Version     : See phx_api.h for overall library release information.
 * Release Date: See phx_api.h for overall library release information.
 *
 * Copyright (c) 2000-2004 Active Silicon Ltd.
 ****************************************************************************
 * Comments:
 * --------
 * This file contains sections for code for each of the operating system
 * environments & driver layer that it supports (e.g. Win32 & CDA driver).
 *
 * Supported operating environments are:
 * ------------------------------------
 * 1. "_PHX_WIN32"   - Win32 (Windows 98, Windows NT4, Windows 2K/XP).
 * 2. "_PHX_DOS32"   - 32 bit DOS (DJGPP).
 * 3. "_PHX_LINUX"   - Linux
 * 4. "_PHX_MACOSX"  - MacosX (10.2 and above)
 * 5. "_PHX_VXWORKS" - VxWorks (? and above)
 * 6. "_PHX_QNX"     - QNX 6.5 and above
 *
 * The sections common to all operating systems are:
 * 1. Malloc and Free macros.
 *
 * The sections within each operating environment are:
 * 1. Include files.
 * 3. General type definitions -"ui32" etc definitions.
 * 4. Library export and calling convention definitions.
 * 5. Error print macros - for pop-up window / printf etc.
 * 6. Assertion macros (included if _PHX_DEBUG defined).
 * 7. Debug macros.
 *
 ****************************************************************************
 */

#ifndef _PHX_OS_H_
#define _PHX_OS_H_

/* For compatibility */
#if (defined _PHX_WIN32 || defined _PHX_WIN64) && !defined _PHX_WIN
#define _PHX_WIN
#endif

#ifdef _PHX_WIN
#pragma warning(disable : 4032) /* warning C4032: formal parameter 1 has       \
                                   different type when promoted */
#pragma warning(disable : 4055) /* warning C4055: type cast from data pointer  \
                                   to function pointer */
#pragma warning(disable : 4102) /* warning C4102: unreferenced label */
#pragma warning(                                                               \
    disable : 4115) /* warning C4115: named type definition in parentheses */
#pragma warning(                                                               \
    error : 4001) /* Warning C4001: C++ comment is now an error; but this is   \
                   * only displayed if the /Za option ("disabled Microsoft                                        \
                   * extensions") is enabled which then generates HUGE numbers                                                        \
                   * of errors and warnings in "windows.h".  It may be         \
                   * possible not to include this header file in the           \
                   * librarues, but the CDA library would also have to be                                                 \
                   * fixed as this is included for the CDA_Buffer info.                                                                         \
                   */
#endif

#if defined _PHX_WIN
/*===========================================================================*/

/*
_PHX_WIN(ForBrief)
-------------------
*/
#if !defined _PDL_WIN
#define _PDL_WIN
#endif
#if !defined _PBL_WIN
#define _PBL_WIN
#endif

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <conio.h>
#include <ctype.h>
#include <errno.h>
#include <io.h>
#include <malloc.h>
#include <math.h>
#include <memory.h>
#include <stdarg.h> /* Must be before stdio.h for MS Windows */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h> /* windows.h must be included first */

/* 2. Malloc/free macros
 * ---------------------
 */
#define _PHX_Malloc(x) malloc(x)
#define _PHX_Free(x) free(x)
#define _PHX_SleepMs(x) Sleep(x);

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
#define _ASL_TYPEDEFS
typedef signed __int8 i8;
typedef unsigned __int8 ui8;
typedef signed __int16 i16;
typedef unsigned __int16 ui16;
typedef signed __int32 i32;
typedef unsigned __int32 ui32;
typedef signed __int64 i64;
typedef unsigned __int64 ui64;
typedef signed __int32 m16;
typedef unsigned __int32 mu16;
typedef signed __int32 m32;
typedef unsigned __int32 mu32;
#if defined _WIN64
typedef unsigned __int64 ptr_t;
#else
typedef unsigned __int32 ptr_t;
#endif
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 * For a DLL these definitions refer to exported data hence, dllexport.
 * However, where they're used they are imported, hence dllimport.
 */
#define PHX_INLINE static __forceinline

#define PHX_C_CALL __cdecl

#if defined _PHX_USEDLLS
#define PHX_EXPORT_FN __declspec(dllimport) PHX_C_CALL
#define PHX_EXPORT_PFN __declspec(dllimport) * PHX_C_CALL
#define PHX_EXPORT_DATA __declspec(dllimport)
#else
#define PHX_EXPORT_FN __declspec(dllexport) PHX_C_CALL
#define PHX_EXPORT_PFN __declspec(dllexport) * PHX_C_CALL
#define PHX_EXPORT_DATA __declspec(dllexport)
#endif

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#ifdef __cplusplus
#define _PHX_ErrPrint(szMessage, szTitle)                                      \
    ::MessageBox(NULL, szMessage, szTitle, MB_OK);
#else
#define _PHX_ErrPrint(szMessage, szTitle)                                      \
    MessageBox(NULL, szMessage, szTitle, MB_OK);
#endif

/* 6. Assert Macros
 * ----------------
 * The Microsoft ANSI C "assert" is not compiled in when NDEBUG is selected,
 * which is selected by default on the project makefiles for release builds.
 */
#define _PHX_Assert assert

#ifdef _PHX_DEBUG
#define _PHX_ErrorTrap(_f, _s)                                                 \
    _PHX_Assert(_f);                                                           \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#else
#define _PHX_ErrorTrap(_f, _s)                                                 \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#endif

/* 7. Debug Macros
 * ---------------
 * _PHX_DebugTrace : Outputs a debug string. If it's a console app, the
 * string is simply printed in the console window. Fow a windows app,
 * a popup is generated.
 */
#ifdef _CONSOLE /* Win32 automatic definition */

#ifdef _PHX_DEBUG

#define _PHX_DebugTrace(_str)                                                  \
    { printf("%s", _str); }

#else /* no debugging - _PHX_DEBUG not defined */

#define _PHX_DebugTrace(_str)

#endif /* #else part of "#ifdef _PHX_DEBUG" */

#else /* #else part of "#ifdef _CONSOLE"   */

#ifdef _PHX_DEBUG

extern ui32 gdwDbgCtrl;

#ifdef __cplusplus
#define _PHX_DebugTrace(_str)                                                  \
    {                                                                          \
        if (PHX_DEBUG_POPUP == (gdwDbgCtrl & PHX_DEBUG_POPUP)) {               \
            ::MessageBox(NULL, _str, "Phoenix Debug Trace", MB_OK);            \
        } else {                                                               \
            OutputDebugString(_str);                                           \
        }                                                                      \
    };
#else
#define _PHX_DebugTrace(_str)                                                  \
    {                                                                          \
        if (PHX_DEBUG_POPUP == (gdwDbgCtrl & PHX_DEBUG_POPUP)) {               \
            MessageBox(NULL, _str, "Phoenix Debug Trace", MB_OK);              \
        } else {                                                               \
            OutputDebugString(_str);                                           \
        }                                                                      \
    };
#endif

#else /* no debugging - _PHX_DEBUG not defined */

#define _PHX_DebugTrace(_str)

#endif /* #else part of "#ifdef _PHX_DEBUG" */

#endif /* #else part of "#ifdef _CONSOLE"  */

#elif defined _PHX_DOS32 /*================================================*/

/*
_PHX_DOS32(ForBrief)
--------------------
*/
#if !defined _CDA_DOS32
#define _CDA_DOS32
#endif
#if !defined _PDL_GRX
#define _PDL_GRX
#endif
#if !defined _DOS32
#define _DOS32
#endif

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <bios.h>
#include <conio.h>
#include <ctype.h>
#include <dos.h>
#include <errno.h>
#include <io.h>
#include <malloc.h>
#include <math.h>
#include <memory.h>
#include <stdarg.h> /* must be before stdio.h for MS Windows */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#define _PHX_Malloc(x) malloc(x)
#define _PHX_Free(x) free(x)
/* The usleep function accepts a value in microseconds.
 * It has a granularity of 55msec but is better than 'sleep'
 */
#define _PHX_SleepMs(x) usleep((x)*1000);

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
#define _ASL_TYPEDEFS
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
typedef unsigned int ptr_t;
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
/* Required to get gcc to compile */
#define PHX_INLINE static __inline__
#define PHX_EXPORT_FN
#define PHX_EXPORT_PFN
#define PHX_EXPORT_DATA

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _PHX_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _PHX_Assert assert

#ifdef _PHX_DEBUG
#define _PHX_ErrorTrap(_f, _s)                                                 \
    _PHX_Assert(_f);                                                           \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#else
#define _PHX_ErrorTrap(_f, _s)                                                 \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#endif

/* 7. Debug Macros
 * ---------------
 * _PHX_DebugTrace : Output a debug string.
 */

#ifdef _PHX_DEBUG

#define _PHX_DebugTrace(_str)                                                  \
    { printf("%s", _str); }

#else /* no debugging - _PHX_DEBUG not defined */

#define _PHX_DebugTrace(_str)

#endif /* #else part of "#ifdef _PHX_DEBUG" */

#elif defined _PHX_LINUX /*================================================*/

/*
_PHX_LINUX(ForBrief)
--------------------
*/

#if !defined _CDA_LINUX
#define _CDA_LINUX
#endif
#if !defined _PDL_XWIN
#define _PDL_XWIN
#endif
#if !defined _PHX_POSIX
#define _PHX_POSIX
#endif

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdarg.h> /* must be before stdio.h for MS Windows */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> /* struct timeval & gettimeofday */
#include <time.h>
#include <unistd.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#define _PHX_Malloc(x) malloc(x)
#define _PHX_Free(x) free(x)
#define _PHX_SleepMs(x)                                                        \
    {                                                                          \
        struct timespec tReq;                                                  \
        tReq.tv_sec = (x) / 1000L;                                             \
        tReq.tv_nsec = ((x) % 1000L) * 1000000UL;                              \
        (void)nanosleep(&tReq, NULL);                                          \
    }

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
#define _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
typedef int i32;
typedef unsigned int ui32;
typedef long m16;
typedef unsigned long mu16;
typedef long m32;
typedef unsigned long mu32;
#if defined _LP64
typedef unsigned long ui64;
typedef unsigned long ptr_t;
#else
typedef unsigned long long ui64;
typedef unsigned int ptr_t;
#endif
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
/* Required to get gcc to compile */
#define PHX_C_CALL
#define PHX_INLINE static __inline__
#define PHX_EXPORT_FN __attribute__((__visibility__("default")))
#define PHX_EXPORT_PFN *PHX_EXPORT_FN
#define PHX_EXPORT_DATA __attribute__((__visibility__("default")))

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _PHX_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _PHX_Assert assert

#ifdef _PHX_DEBUG
#define _PHX_ErrorTrap(_f, _s)                                                 \
    _PHX_Assert(_f);                                                           \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#else
#define _PHX_ErrorTrap(_f, _s)                                                 \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#endif

/* 7. Debug Macros
 * ---------------
 * _PHX_DebugTrace : Output a debug string.
 */

#ifdef _PHX_DEBUG

#define _PHX_DebugTrace(_str)                                                  \
    { printf("%s", _str); }

#else /* no debugging - _PHX_DEBUG not defined */

#define _PHX_DebugTrace(_str)

#endif /* #else part of "#ifdef _PHX_DEBUG" */

/* 8. Additional Controls
 * ----------------------
 */

#elif defined _PHX_MACOSX /*================================================*/

/*
_PHX_MACOSX(ForBrief)
--------------------
*/
#if !defined _CDA_MACOSX
#define _CDA_MACOSX
#endif
#if !defined _PDL_MACOSX
#define _PDL_MACOSX
#endif
#if !defined _PHX_POSIX
#define _PHX_POSIX
#endif
#if !defined _PHX_BIG_ENDIAN && defined(__BIG_ENDIAN__)
#define _PHX_BIG_ENDIAN
#endif

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdarg.h> /* Must be before stdio.h for MS Windows */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> /* struct timeval & gettimeofday */
#include <time.h>
#include <unistd.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#define _PHX_Malloc(x) malloc(x)
#define _PHX_Free(x) free(x)
#define _PHX_SleepMs(x) usleep((x)*1000);

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
#define _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef int16_t i16;
typedef uint16_t ui16;
typedef int32_t i32;
typedef uint32_t ui32;
typedef uint64_t ui64;
typedef int m16;
typedef unsigned int mu16;
typedef int m32;
typedef unsigned int mu32;
typedef unsigned long ptr_t;
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define PHX_INLINE static __inline__
#define PHX_EXPORT_FN
#define PHX_EXPORT_PFN *
#define PHX_EXPORT_DATA

#define PHX_C_CALL

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _PHX_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _PHX_Assert assert

#ifdef _PHX_DEBUG
#define _PHX_ErrorTrap(_f, _s)                                                 \
    _PHX_Assert(_f);                                                           \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#else
#define _PHX_ErrorTrap(_f, _s)                                                 \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#endif

/* 7. Debug Macros
 * ---------------
 */
#ifdef _PHX_DEBUG

#define _PHX_DebugTrace(_str)                                                  \
    { printf("%s", _str); }
/* _CDA_Debug: Outputs a long word as part of a debug string.
   ---------- */
/*#define _CDA_Debug(pszMessage, dwParameter1, dwParameter2) \
  { printf("CDA_Debug (%s) - %s: 0x%02lx (%02ld) : 0x%02lx (%02ld)\n", szFnName,
  pszMessage, (ui32) dwParameter1, (ui32) dwParameter1, (ui32) dwParameter2,
  (ui32) dwParameter2); }
 */

/* _CDA_DebugString: Outputs a string as part of a debug string.
   --------------- */
/*#define _CDA_DebugString(pszMessage, pszParameter) { \
  char _szWork[CDA_DEBUG_STRING_LEN]; char _szParamString[CDA_DEBUG_STRING_LEN];
  strcpy(_szWork, szFnName); \
  strcat(_szWork, " - "); strcat(_szWork, pszMessage); \
  sprintf(_szParamString, " %s\r", (ui32) pszParameter); \
  strcat(_szWork, _szParamString); \
  printf( _szWork ); }
 */

/* _CDA_DebugPopup: Outputs a debug string in a popup.
   -------------- */
/*#ifdef __cplusplus
#define _CDA_DebugPopup(pszMessage, pszTitle)
#else
#define _CDA_DebugPopup(pszMessage, pszTitle)
#endif
*/
#else /* no debugging - _PHX_DEBUG not defined */

#define _PHX_DebugTrace(_str)
/*
#define _CDA_Debug(pszMessage, dwParameter1, dwParameter2)
#define _CDA_DebugString(pszMessage, pszParameter)
#define _CDA_DebugPopup(pszMessage, pszTitle)
*/
#endif /* #else part of "#ifdef _PHX_DEBUG" */

/* 8. Additional Controls
 * ----------------------
 */

#elif defined _PHX_VXWORKS /*================================================*/

/*
_PHX_VXWORKS(ForBrief)
-------------------
*/
#if !defined _CDA_VXWORKS
#define _CDA_VXWORKS
#endif

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <semLib.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <types/vxCpu.h>

#if (CPU_FAMILY == PPC)
#if !defined _PHX_BIG_ENDIAN
#define _PHX_BIG_ENDIAN
#endif
#elif (CPU_FAMILY == I80X86)
#if defined _PHX_BIG_ENDIAN
#undef _PHX_BIG_ENDIAN
#endif
#else
#error Unrecognised VxWorks CPU definition
#endif

/* 2. Malloc/free macros
 * ---------------------
 */
#define _PHX_Malloc(x) malloc(x)
#define _PHX_Free(x) free(x)
#define _PHX_SleepMs(x) CDA_DRV_Delay_ms(0, (x))

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
#define _ASL_TYPEDEFS
typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
typedef int i32;
typedef unsigned int ui32;
typedef unsigned long long ui64;
typedef int m16;
typedef unsigned int mu16;
typedef int m32;
typedef unsigned int mu32;
typedef unsigned int ptr_t;
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
#define PHX_INLINE static __inline__
#define PHX_EXPORT_FN
#define PHX_EXPORT_PFN *
#define PHX_EXPORT_DATA

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _PHX_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _PHX_Assert assert

#ifdef _PHX_DEBUG
#define _PHX_ErrorTrap(_f, _s)                                                 \
    _PHX_Assert(_f);                                                           \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#else
#define _PHX_ErrorTrap(_f, _s)                                                 \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#endif

/* 7. Debug Macros
 * ---------------
 * _PHX_DebugTrace : Output a debug string.
 */

#ifdef _PHX_DEBUG
#define _PHX_DebugTrace(_str)                                                  \
    { printf("%s", _str); }

#else /* no debugging - _PHX_DEBUG not defined */
#define _PHX_DebugTrace(_str)

#endif /* #else part of "#ifdef _PHX_DEBUG" */

#elif defined _PHX_QNX /*================================================*/

/*
_PHX_QNX(ForBrief)
--------------------
*/

#if !defined _CDA_QNX
#define _CDA_QNX
#endif
#if !defined _PHX_POSIX
#define _PHX_POSIX
#endif

/* 1. Include Files
 * ----------------
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/neutrino.h>
#include <sys/time.h> /* struct timeval & gettimeofday */
#include <time.h>
#include <unistd.h>

/* 2. Malloc/free macros
 * ---------------------
 */
#define _PHX_Malloc(x) malloc(x)
#define _PHX_Free(x) free(x)
#define _PHX_SleepMs(x)                                                        \
    {                                                                          \
        struct timespec tReq;                                                  \
        tReq.tv_sec = (x) / 1000;                                              \
        tReq.tv_nsec = ((x) % 1000) * 1000000;                                 \
        (void)nanosleep(&tReq, NULL);                                          \
    }

/* 3. General type definitions
 * ---------------------------
 */
#ifndef _ASL_TYPEDEFS
#define _ASL_TYPEDEFS
typedef int8_t i8;
typedef uint8_t ui8;
typedef int16_t i16;
typedef uint16_t ui16;
typedef int32_t i32;
typedef uint32_t ui32;
typedef int32_t m16;
typedef uint32_t mu16;
typedef int32_t m32;
typedef uint32_t mu32;
typedef uint32_t ptr_t;
typedef uint64_t ui64;
/*   typedef uint32_t tFlag;*/
#endif

/* 4. Library export and calling convention definitions
 * ----------------------------------------------------
 */
/* Required to get gcc to compile */
#define PHX_C_CALL
#define PHX_INLINE static __inline__
#define PHX_EXPORT_FN __attribute__((__visibility__("default")))
#define PHX_EXPORT_PFN *PHX_EXPORT_FN
#define PHX_EXPORT_DATA __attribute__((__visibility__("default")))

/* 5. Error handler macro - used by default error handler
 * ------------------------------------------------------
 */
#define _PHX_ErrPrint(szMessage, szTitle)                                      \
    { printf("\nERROR - %s: %s\n", szTitle, szMessage); }

/* 6. Assert Macros
 * ----------------
 */
#define _PHX_Assert assert

#ifdef _PHX_DEBUG
#define _PHX_ErrorTrap(_f, _s)                                                 \
    _PHX_Assert(_f);                                                           \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#else
#define _PHX_ErrorTrap(_f, _s)                                                 \
    if (!(_f))                                                                 \
        _PHX_GotoErr(PHX_ERROR_INTERNAL_ERROR, _s);
#endif

/* 7. Debug Macros
 * ---------------
 * _PHX_DebugTrace : Output a debug string.
 */

#ifdef _PHX_DEBUG

#define _PHX_DebugTrace(_str)                                                  \
    { printf("%s", _str); }

#else /* no debugging - _PHX_DEBUG not defined */

#define _PHX_DebugTrace(_str)

#endif /* #else part of "#ifdef _PHX_DEBUG" */

/* 8. Additional Controls
 * ----------------------
 */

#else

#error You have a missing target environment directive (e.g. _PHX_WIN32)

#endif

/* Common Definitions for All OS's
 * ===============================
 */
/* 1. Common Constants
 * -------------------
 */

#ifndef TRUE
#define TRUE (1 == 1)
#define FALSE (!TRUE)
#endif

/* Type Definitions
 * ================
 */
typedef mu32 tHandle;
typedef ui32 tFlag;
typedef const char *const tPhxFnName;
typedef const char *tPhxErrStr;
typedef void *tPHX;

typedef ui64 tPhxPhysAddr;
typedef ui64 tPhxPhysLen;

#endif /* _PHX_OS_H_ */
