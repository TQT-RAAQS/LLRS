/****************************************************************************
*
* ACTIVE SILICON LIMITED
*
* File name   : as_trace_error.h
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
* \file as_trace_error.h
* \addtogroup TraceLibrary
* \{
*/

#ifndef _AS_TRACE_ERROR_H_
#define _AS_TRACE_ERROR_H_

typedef enum AS_TraceError {
     AS_TraceNoError             /*! Function succeeded. */
   , AS_TraceInvalidArgument     /*! One of more arguments passed to the function have invalid value(s). */
   , AS_TraceRegistryError       /*! Error whilst accessing the Windows registry. */
   , AS_TraceInitialisedMaxCountReached  /*! The library has been initialized too many times (current max count is 255). */
   , AS_TraceAlreadyInitialised  /*! The library is already initialized and is ready to be used. */
   , AS_TraceNotInitialised      /*! The library is not yet initialized; TraceOpen() should be called first. */
   , AS_TraceIOError             /*! Error accessing disk. */
   , AS_TraceInternalError       /*! An internal error. */
   , AS_TraceOutOfMemory         /*! Machine has run out of memory. */
} AS_etTraceError;

#endif /* _AS_TRACE_ERROR_H_ */

/*! \}*/
