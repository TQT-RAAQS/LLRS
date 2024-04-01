/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : tmg_err.h
 * Function    : Status and error definitions.
 * Project     : General
 * Authors     : Colin Pearce
 * Systems     : ANSI C, various.
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
 *
 ****************************************************************************
 */

#ifndef _TMG_ERR_H_
#define _TMG_ERR_H_

/*
 * Status and Error Return Codes
 * -----------------------------
 * Status values are returned as a 32 bit integer with the top 16 bits
 * indicating where the error came from (top 8 bits) and the actual error
 * code (next 8 bits down). The lower 16 bits may be used for any other
 * purpose - such as to return more detailed status information etc.
 *
TMG_Status_and_Error_Return_Codes(ForBrief)
 */
#define TMG_MAX_ERROR_STRING_LENGTH 512 /* Used in TMG_ErrDisplay */
#define TMG_ERROR_ID_CODE ((ui32)TMG_MAGIC_ERROR)
#define TMG_ERROR_CODE_MASK ((ui32)0xFFFF0000)

#define TMG_OK ((ui32)0)
#define TMG_ERROR_NOT_TMG ((ui32)TMG_ERROR_ID_CODE | 1 << 16)
#define TMG_ERROR_INTERNAL ((ui32)TMG_ERROR_ID_CODE | 2 << 16)
#define TMG_ERROR_BAD_PARAM ((ui32)TMG_ERROR_ID_CODE | 3 << 16)
#define TMG_ERROR_BAD_HANDLE ((ui32)TMG_ERROR_ID_CODE | 4 << 16)
#define TMG_ERROR_BAD_IMAGE ((ui32)TMG_ERROR_ID_CODE | 5 << 16)
#define TMG_ERROR_OUT_OF_MEMORY ((ui32)TMG_ERROR_ID_CODE | 6 << 16)
#define TMG_ERROR_NOT_SUPPORTED ((ui32)TMG_ERROR_ID_CODE | 7 << 16)
#define TMG_ERROR_OUT_OF_HANDLES ((ui32)TMG_ERROR_ID_CODE | 8 << 16)
#define TMG_ERROR_OPEN_FAILED ((ui32)TMG_ERROR_ID_CODE | 9 << 16)
#define TMG_ERROR_CLOSE_FAILED ((ui32)TMG_ERROR_ID_CODE | 10 << 16)
#define TMG_ERROR_INVALID_STATE ((ui32)TMG_ERROR_ID_CODE | 11 << 16)
#define TMG_ERROR_UNEXPECTED ((ui32)TMG_ERROR_ID_CODE | 12 << 16)
#define TMG_ERROR_SYS_CALL_FAILED ((ui32)TMG_ERROR_ID_CODE | 13 << 16)

/*
 * Status and Error Macros
 * -----------------------
 * _TMG_Roe    - Tests result and returns result if it is an error.
 * _TMG_Proe   - Tests result, if it is an error, the installed error handler
 *               is called.  The default error handler (TMG_ErrorHandler)
 *               will print a message.
 *               WARNING: Use these macros with "{}" in an "else" statement.
 * _TMG_ErrRet - Used for returning an error with an additional 16 bit
 *               error value (often used to represent the Nth occurrence
 *               in a function).
 *
 * NOTE: "hLib" is forced to 1. Multi-use instances of a shared library
 * has to use common error handlers (for now).
 *
Status_and_Error_Macros(ForBrief)
 */

/* _TMG_Roe()  --Threadsafe-- */
#define _TMG_Roe(Function)                                                     \
    {                                                                          \
        ui32 _dwStatus;                                                        \
        _dwStatus = Function;                                                  \
        if ((_dwStatus & TMG_ERROR_CODE_MASK) != 0)                            \
            return (_dwStatus);                                                \
    }

/* _TMG_Proe()  --Threadsafe-- */
#define _TMG_Proe(Function, szDescString)                                      \
    {                                                                          \
        ui32 _dwStatus;                                                        \
        _dwStatus = Function;                       /*lint -save -e774 */      \
        if ((_dwStatus & TMG_ERROR_CODE_MASK) != 0) /*lint -restore */         \
        {                                                                      \
            TMG_ErrProcess(1, szFnName, _dwStatus, szDescString);              \
            return (_dwStatus);                                                \
        }                                                                      \
    }

/* _TMG_ErrRet()  --Threadsafe-- */
#define _TMG_ErrRet(dwErrCode, szDescString, dwReturnValue)                    \
    {                                                                          \
        ui32 _dwStatus;                                                        \
        _dwStatus = dwErrCode | (dwReturnValue & 0xFFFF);                      \
        TMG_ErrProcess(1, szFnName, _dwStatus, szDescString);                  \
        return (_dwStatus);                                                    \
    }

#endif /* _TMG_ERR_H_ */
