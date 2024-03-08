/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : pcc_api.h
 * Function    : Header file for CPcc class
 * Project     : AP06
 * Authors     : Richard Brown
 * Systems     : Win32 (MSVC).
 * Version     : 1.0
 * Release Date: 19-Sep-02
 *
 * Copyright (c) 2002 Active Silicon Ltd.
 ****************************************************************************
 * Comments:
 * --------
 * This class defines base class for modeless property sheet CPhxCtrl
 *
 ****************************************************************************
 */
 
#ifndef __PCC_API_H__
#define __PCC_API_H__

#include <phx_api.h>

// --- dwPagesVisible
#define PCC_CONFIG      0x00000001
#define PCC_DESTINATION 0x00000002
#define PCC_CAMERA      0x00000004
#define PCC_ROI         0x00000008
#define PCC_LUT         0x00000010
#define PCC_ACQTRIG     0x00000020
#define PCC_COMMS       0x00000040
#define PCC_STATUS      0x00000080
#define PCC_TEST        0x00000100
#define PCC_IOPORTS     0x00000200
#define PCC_INTRPT      0x00000400
#define PCC_EXPOSURE    0x00000800
#define PCC_POCL        0x00001000
#define PCC_DEVICE      0x00002000
#define PCC_CXP         0x00004000
#define PCC_FBDIO       0x00008000
#define PCC_CXP_ERRORS  0x00010000
#define PCC_DEFAULT     0x00000EFF

void CtrlEventCallback( tHandle m_hCamera, ui32 dwInterruptMask, void *vParams );

/////////////////////////////////////////////////////////////////////////////
// CPhxCtrl

typedef const enum {
   PCC_EMASK_FN         = (int) ((ui32) FNTYPE_EMASK   | (ui32) 0x000F0000),
   PCC_SETANDGET        = (int) ((ui32) FNTYPE_PCC_API | (ui32) 0x00010000),
   PCC_EXEC             = (int) ((ui32) FNTYPE_PCC_API | (ui32) 0x00020000)
} etPccFn ;

typedef enum
{
   PCC_CAM_HANDLE    = (int) ((ui32) PCC_SETANDGET | ((ui32)  1 << 8 )),
   PCC_EVENT_HANDLER = (int) ((ui32) PCC_SETANDGET | ((ui32)  2 << 8 )),
   PCC_GET_SAFE_HWND = (int) ((ui32) PCC_SETANDGET | ((ui32)  3 << 8 )),
   PCC_SHOW_WINDOW   = (int) ((ui32) PCC_SETANDGET | ((ui32)  4 << 8 )),
   PCC_CURRENT_IMAGE = (int) ((ui32) PCC_SETANDGET | ((ui32)  5 << 8 ))
} etPccParam;

typedef enum
{
   PCC_CREATE          = (int) ((ui32) PCC_EXEC | ((ui32)  1 << 8 )),
   PCC_MESSAGE_HANDLER = (int) ((ui32) PCC_EXEC | ((ui32)  2 << 8 ))
} etPccExec;



class __declspec(dllexport) CPcc
{
   // Construction
public:
   
   void *m_pPccCtrl;    // --- Pointer to CPhxCtrl object

   CPcc( CWnd* pParentView, void (*pFnErrorHandler) (const char *, etStat, const char *) );   
	CPcc( CWnd* pParentView, unsigned long dwPagesVisible, void (*pFnErrorHandler) (const char *, etStat, const char *) );
   CPcc( CWnd* pParentView, unsigned long dwPagesVisible, tFlag fPushDocTitle, void (*pFnErrorHandler) (const char *, etStat, const char *) );
   ~CPcc();

   etStat   Exec            ( etPccExec );
   etStat   ParameterGet    ( etPccParam, void* );
   etStat   ParameterSet    ( etPccParam, void* );
   void     DisplayStats    ( void );
   void     PrintToStatusBar( char *szMessage );
   void     SaveSequence    ( tHandle hPhx, ui32 dwStart, ui32 dwEnd );
};

BOOL __declspec(dllexport) FilterDllMsg(LPMSG lpMsg);

/////////////////////////////////////////////////////////////////////////////

#endif	// __PCC_API_H__
