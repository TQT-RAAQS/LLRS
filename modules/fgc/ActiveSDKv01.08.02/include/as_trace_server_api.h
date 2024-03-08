/****************************************************************************
*
* ACTIVE SILICON LIMITED
*
* File name   : as_trace_server_api.h
* Function    : User API for the Active Silicon Trace Server library
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
* \file as_trace_server_api.h
*/

#ifndef _AS_TRACE_SERVER_API_H_
#define _AS_TRACE_SERVER_API_H_

#include <as_trace_os.h>
#include <as_trace_error.h>

#ifdef __cplusplus
extern "C" {
#endif

   /*!
   * \defgroup TraceServer Trace Server
   * \{
   * \brief The Active Silicon Trace Server library allows to control the Trace library from
   * a remote process. The implementation uses a client / server architecture running over IP.
   * 
   * Example:
   * 
   * \code
   * #include <as_trace_api.h>
   * 
   * int main()
   * {
   *     if (AS_TraceServerStart() != AS_TraceNoError) {
   *        std::cout << "Failed starting server" << std::endl;
   *        return EXIT_FAILURE;
   *     }
   *     
   *     ... your application-specific code here ...
   *     
   *     // Generate some trace events using the AS Trace library
   *     AS_TraceF("My event");
   *     
   *     ... more application-specific code here ...
   *     
   *     if (AS_TraceServerStop() != AS_TraceNoError) {
   *        std::cout << "Failed stopping server" << std::endl;
   *        return EXIT_FAILURE;
   *     }
   *     
   *     return 1;
   * }
   * \endcode
   */

#if !defined _TRACE_LOAD_MANUAL

   /*!
   * \brief Initialise the Trace library (AS_TraceOpen()) and start the Trace server.
   * There can only be one Trace server, so the library maintains an internal reference count,
   * so calling this function more than once succeeds and increments the reference count. 
   * AS_TraceServerStop() must called as many times as AS_TraceServerStart() to ensure the Trace
   * server stops.
   * 
   * Calling this function starts the Trace RPC server and the Trace RPC Registration client:
   * - The Trace RPC server waits for requests from the process that wishes to interact with the 
   * Trace library (this is typically a GUI provided by Active Silicon).
   * - The Trace RPC Registration client is part of the infrastructure and is there to establish
   * and maintain the connection with the process that wishes to interact with the Trace
   * library (this is typically a GUI provided by Active Silicon).
   *
   * The function internally creates threads to run the Trace server loop.
   */
   AS_etTraceError AS_TRACE_SERVER_EXPORT_FN AS_TraceServerStart();

   /*!
    * \brief Stop the Trace server (also calls AS_TraceClose() to deinitialise Trace library).
   */
   AS_etTraceError AS_TRACE_SERVER_EXPORT_FN AS_TraceServerStop();

#else

extern AS_etTraceError(*AS_TraceServerStart)();
extern AS_etTraceError(*AS_TraceServerStop)();

#endif //_TRACE_LOAD_MANUAL

#ifdef __cplusplus
}
#endif

#endif /* _AS_TRACE_SERVER_API_H_ */

/*! \}*/