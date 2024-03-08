#ifndef _AS_TRACE_OS_H_
#define _AS_TRACE_OS_H_

#if defined (_MSC_VER)
#define AS_TRACE_SERVER_CALL_CONV __cdecl

   #if defined AS_TRACE_SERVER_EXPORT
      #define AS_TRACE_SERVER_DECLSPEC __declspec(dllexport)
   #else
      #define AS_TRACE_SERVER_DECLSPEC __declspec(dllimport)
   #endif
#else
   #define AS_TRACE_SERVER_CALL_CONV
   #define AS_TRACE_SERVER_DECLSPEC __attribute__((__visibility__("default")))
#endif

#define AS_TRACE_SERVER_EXPORT_FN AS_TRACE_SERVER_DECLSPEC AS_TRACE_SERVER_CALL_CONV

#endif /* _AS_TRACE_OS_H_ */
