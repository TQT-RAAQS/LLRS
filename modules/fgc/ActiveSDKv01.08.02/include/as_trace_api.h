/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : as_trace_api.h
 * Function    : User API for the Active Silicon Trace Library
 * Project     :
 * Authors     : Jean-Philippe Arnaud
 * Systems     : ANSI C
 *
 * Copyright (c) 2000 - 2018 Active Silicon Ltd.
 ****************************************************************************
 * Comments:
 * --------
 *
 ****************************************************************************
 */

/*!
 * \file as_trace_api.h
 */

#ifndef _AS_TRACE_API_H_
#define _AS_TRACE_API_H_

#if defined(_MSC_VER)
#define CALLING_CONVENTION __cdecl

#if defined TRACE_EXPORT
#define EXPORT_FN __declspec(dllexport)
#else
#define EXPORT_FN __declspec(dllimport)
#endif
#else
#define CALLING_CONVENTION
#define EXPORT_FN __attribute__((__visibility__("default")))
#endif

#define TRACE_EXPORT_FN EXPORT_FN CALLING_CONVENTION

#include "as_trace_error.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \defgroup TraceLibrary Trace Library
 * \{
 * \brief The Trace library provides an API to log messages when an application
 * is running. It is a debug tool that may be used to understand the root cause
 * of a problem or to optimise an application. It is used by Active Silicon's
 * own libraries but can also be used from user applications.
 *
 *        The library is a shared object (DLL or SO) and manages a single
 * instance of a circular memory buffer in which the log messages are inserted.
 *
 *        The library streams the content of the buffer to disk on request.
 *
 *        For convenience, the parameters used to control the Trace are read
 * from a database at runtime. This allows changing the parameters without
 * rebuilding the application. Under Windows, this is implemented via the
 * registry and a GUI is provided to allow changing the parameters. Under Linux,
 * this is implemented via a configuration file: $HOME/.as_conf.cfg.
 *
 *        Each log message is prefixed with a standard prefix that includes the
 * timestamp, the thread ID and priority.
 *
 *        Under Windows, the content of the circular buffer is automatically
 * saved to a temporary directory when the process detaches from the library.
 * Under Windows this is the Temporary directory (environment variable TEMP).
 * The file is overwritten each time so that disk usage is limited.
 *
 * \section UsingTheTraceLibrary Using the Trace Library
 *        Link your application against the library, either using the import
 * library supplied or by dynamically loading the library at runtime, in which
 * case you may want to define _TRACE_LOAD_MANUAL.
 *
 *        Include the library's header file and use function AS_TraceF() to
 * insert log messages. Call function AS_TraceSave() to stream the log to disk.
 */

/*!
 * \brief The maximum length or a message; the library can be configured to
 * create shorter messages via the UI, but this is the absolute maximum length
 * possible.
 */
#define AS_kTraceMaxMsgLength 1024

#define AS_TraceApiVersion_1_0_0 0x00010000

#if !defined _TRACE_LOAD_MANUAL
/*!
 * \brief This function must be called once prior to using any other functions.
 *        The library does reference counting so there should be a matching
 * number of calls to AS_TraceClose(). The global trace instance is initialized
 * upon the return of the function.
 */
AS_etTraceError TRACE_EXPORT_FN AS_TraceOpen();

/*!
 * \brief Free the global trace instance if the reference has reached 0.
 * No other function should be called after this call except for AS_TraceOpen().
 */
AS_etTraceError TRACE_EXPORT_FN AS_TraceClose();

/*!
 * \brief Formatted Trace. Insert formatted string into the trace. The standard
 * prefix is automatically added. \param fmt C string that contains the text to
 * be inserted. It can optionally contain embedded format specifiers that are
 * replaced by the values specified in subsequent additional arguments and
 * formatted as requested.
 */
AS_etTraceError TRACE_EXPORT_FN AS_TraceF(const char *fmt, ...);

/*!
 * \brief Same as AS_TraceF(), but the standard prefix's timestamp is provided
 * by caller. \param fmt C string that contains the text to be inserted. It can
 * optionally contain embedded format specifiers that are replaced by the values
 * specified in subsequent additional arguments and formatted as requested.
 */
AS_etTraceError TRACE_EXPORT_FN AS_TraceTimestampF(uint64_t qwTimeStamp,
                                                   const char *fmt, ...);

/*!
 * \brief Streams the content of the trace to the file selected via the UI.
 *        The file location is retrieved from the the Windows registry each time
 *        the function is called. If no location has been set by the user, the
 * log file is created in the current working directory. The filename is
 * suffixed with the process ID so that multiple processes do not attempt to
 * access the same file at the same time. A CSV file is generated, with a header
 * describing its content. \param szRootName Optional root file name. \param
 * szOutFile Optionally the function returns the full path to the file created.
 *                  May be set to NULL. Must be freed using AS_TraceFree().
 */
AS_etTraceError TRACE_EXPORT_FN AS_TraceSave(const char *szRootName,
                                             char **szOutFile);

/*!
 * \brief Free memory pointed to by p. The memory must have been allocated by
 * the Trace library. \param p Memory to free.
 */
void TRACE_EXPORT_FN AS_TraceFree(void *p);

/*!
 * \brief Return non zero is trace is enabled, zero otherwise.
 */
int TRACE_EXPORT_FN AS_TraceIsEnabled();

/*!
 * \brief Return a human-readable representation for the error code \e.
 */
EXPORT_FN const char *CALLING_CONVENTION
AS_TraceErrorToString(AS_etTraceError e);

/*!
 * \brief Return the version of the API implemented by this library. This
 * information allows an application to know which functions are available.
 */
uint32_t TRACE_EXPORT_FN AS_TraceGetApiVersion();

#else

typedef AS_etTraceError (*AS_TraceServerStart_t)();
typedef AS_etTraceError (*AS_TraceServerStop_t)();

typedef AS_etTraceError (*AS_TraceOpen_t)();
typedef AS_etTraceError (*AS_TraceClose_t)();
typedef AS_etTraceError (*AS_TraceF_t)(const char *fmt, ...);
typedef AS_etTraceError (*AS_TraceTimestampF_t)(uint64_t qwTimeStamp,
                                                const char *fmt, ...);
typedef AS_etTraceError (*AS_TraceSave_t)(const char *szRootName,
                                          char **szOutFile);
typedef void (*AS_TraceFree_t)(void *p);
typedef int (*AS_TraceIsEnabled_t)();
typedef const char *(*AS_TraceErrorToString_t)(AS_etTraceError e);
typedef uint32_t (*AS_TraceGetApiVersion_t)();

extern AS_TraceServerStart_t AS_TraceServerStart;
extern AS_TraceServerStop_t AS_TraceServerStop;
extern AS_TraceOpen_t AS_TraceOpen;
extern AS_TraceClose_t AS_TraceClose;
extern AS_TraceF_t AS_TraceF;
extern AS_TraceTimestampF_t AS_TraceTimestampF;
extern AS_TraceSave_t AS_TraceSave;
extern AS_TraceFree_t AS_TraceFree;
extern AS_TraceIsEnabled_t AS_TraceIsEnabled;
extern AS_TraceErrorToString_t AS_TraceErrorToString;
extern AS_TraceGetApiVersion_t AS_TraceGetApiVersion;
#endif //_TRACE_LOAD_MANUAL

#ifdef __cplusplus
}
#endif

/*! \}*/

#endif /* _AS_TRACE_API_H_ */