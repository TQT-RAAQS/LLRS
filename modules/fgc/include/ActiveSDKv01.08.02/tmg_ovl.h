/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : tmg_ovl.h (was as_dsply.h)
 * Function    : Video display functions.
 * Project     : TMG Imaging Library
 * Authors     : Lawrence Rust
 * Systems     : ANSI C, Windows XP (MSVC)
 * Version     : See below.
 *
 * Copyright (c) 2000-2007 Active Silicon Ltd.
 ****************************************************************************
 * File History
 * ------------
 * 30-Oct-00, LVR, v1.00 - Created.
 * 26-Nov-01, LVR, v1.01 - Added ASL_DisplaySetVideoSize.
 * 28-Nov-01, LVR, v1.02 - LFG Win32 Video For Windows driver support
 * 01-Jun-07, HCP, v1.03 - Significant tidying and integration into TMG library.
 * 19-Mar-14, ED,  v1.04 - Rename ASL_SUCCEEDED to ASL_SUCCEEDED_LVR.
 *
 * Comments:
 * --------
 * This file contains the "driver" interface to the low level video display
 * library.  This low level library is wrapped by a set of TMG functions
 * in tmg_ovl_api.c.  The low level functions are in tmg_ovl_drv.c.
 *
 ****************************************************************************/

#ifndef _TMG_OVL_H_
#define _TMG_OVL_H_


#if defined _TMG_WINDOWS
#include <windows.h>
#endif


/* Chroma-Keying Color for Video Overlay:
 */
#define TMG_KEY_RED 0x61
#define TMG_KEY_GRN 0x43
#define TMG_KEY_BLU 0x61

#ifndef ASL_PTR
 #define ASL_PTR( _r) _r *
#endif

#ifndef ASL_FNPTR
 #define ASL_FNPTR( _r) _r *
#endif

#if !defined ASL_DSPLY_EXPORT
#define ASL_DSPLY_EXPORT  //9999
#endif

#ifndef ASL_EXPORT
 #if defined WIN32 && !defined ASL_DSPLY_LIB
  #if defined ASL_DSPLY_EXPORT
   #define ASL_EXPORT( _ret, _fn) extern _ret __declspec( dllexport) _fn
  #else
   #define ASL_EXPORT( _ret, _fn) extern _ret __declspec( dllimport) _fn
  #endif
 #else
  #define ASL_EXPORT( _ret, _fn) extern _ret _fn
 #endif
#endif

/* Make a 'fourcc' code for ASL_DisplayFormat::uFourCC */
#ifndef ASL_MAKEFOURCC
#define ASL_MAKEFOURCC(ch0, ch1, ch2, ch3) \
                ((ui32)(ui8)(ch0) | ((ui32)(ui8)(ch1) << 8) | \
                ((ui32)(ui8)(ch2) << 16) | ((ui32)(ui8)(ch3) << 24 ))
#endif

#define kszRegPath "Software\\ActiveSilicon\\DSPLYW32"

/* Error reporting */
#define DISPLAY_ERROR( _r) (ui32)(ASL_kDisplayError_ ## _r + (__LINE__ * ASL_kDisplayError_Locus))

/* Handle validation prototype */
/*int IsValidDisplay( ASL_DisplayHandle );*/


/* Types */
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Opaque display instance reference */
typedef ASL_PTR( struct ASL_Display) ASL_DisplayHandle;

/* API return codes */
typedef enum ASL_DisplayError
  {
  ASL_kDisplayError_None = 0,

  /* Error locus base (20 bits) - added to error code */
  ASL_kDisplayError_Locus = 0x100,

  /* Error type (3 bits) */
  ASL_kDisplayInfo =  0x10000000,
  ASL_kDisplayWarn =  0x20000000,
  ASL_kDisplayError = 0x40000000 + 0 * ASL_kDisplayError_Locus,

  /* Error code (8 bits) */
  ASL_kDisplayError_NotImplemented = ASL_kDisplayError,
  ASL_kDisplayError_BadParameter,
  ASL_kDisplayError_NoResources,
  ASL_kDisplayError_Device,

  /* Add new errors here */

  ASL_kDisplayError_Max
} ASL_DisplayError;

#ifndef ASL_SUCCEEDED_LVR
 #define ASL_SUCCEEDED_LVR( _u) !(ASL_kDisplayError & (_u))
#endif


/* Video pixel format descriptor */
typedef struct ASL_DisplayFormat
  {
  ui32 uFourCC;                       /* FOURCC code, 0= none */
  ui32 uBitsPerPixels;
  ui32 uRedMask;                      /* or Y, 0 if paletted */
  ui32 uGreenMask;                    /* or U */
  ui32 uBlueMask;                     /* or V */
  } ASL_DisplayFormat;


/* Video buffer descriptor */
typedef struct ASL_DisplayBuffer
  {
  ui32 uFormat;                       /* Index to pixel format array, 0.. */
  ui32 uBuffers;                      /* No. entries in the buffer chain */
  ASL_PTR( void) bufferChain[3];      /* Buffer chain, array of frame buffer ptrs */
  ui32 uStride;                       /* Bytes per line */
  ui32 left, top, right, bottom;      /* Video ROI */
  } ASL_DisplayBuffer;


/* Display option flags */
typedef enum ASL_DisplayFlag
  {
  ASL_kDisplayFlagDirectToScreen    = (1<<0), /* r/w Hardware can display video direct to screen */
  ASL_kDisplayFlagHasOverlay        = (1<<1), /* r/o Graphic/text overlay supported */
  ASL_kDisplayFlagEnableOverlay     = (1<<2), /* r/w */
  ASL_kDisplayFlagBufferChaining    = (1<<3), /* r/w Multiple frames are stored in buffer chains */
  ASL_kDisplayFlagCanStretch        = (1<<4), /* r/o Can stretch video in h/w */
  ASL_kDisplayFlagVideoBobbing      = (1<<5)  /* r/w Stagger odd/even fields for interlacing */
  } ASL_DisplayFlag;


/* ASL_DisplayCreate display info */
#if defined _TMG_WINDOWS
typedef HWND ASL_DisplayInfo;

#elif defined ASL_DISPLY_X
typedef ASL_PTR( const struct ASL_DisplayX
  {
  Display* dpy,                       /* X display */
  Window w,                           /* X window */
  }) ASL_DisplayInfo;

#else
 #error
#endif


/* The display object */
struct ASL_Display
  {
  enum EMagic eMagic;
  CRITICAL_SECTION cs;                /* Write guard */

  /* Video source */
  ui32 uImageWidth;
  ui32 uImageHeight;
  ui32 uBuffers;
  ui32 uFormats;                      /* No. pixel formats */
  const ASL_DisplayFormat* formatArray;/* Pixel formats */
  RECT rectVideo;                     /* Video ROI */

  /* Configuration */
  ui32 flags;                         /* Capabilities & options */
  COLORREF colorBkgrnd;               /* The overlay key colour */
  ui32 format;                        /* Current index into formatArray */

  /* Display area */
  HWND hwnd;
  RECT rectDisplay;

  WNDPROC pfnWndProc;
//#if OPT_PALETTE
  HPALETTE hPalette;
//#endif

#if defined _WINDOWS  /* So that console applications work - they don't have _WINDOWS defined */
  /* DirectDraw objects */
  LPDIRECTDRAW pDD;
  LPDIRECTDRAWSURFACE psfcPrimary;    /* Primary surface */
  LPDIRECTDRAWCLIPPER pClipper;       /* Primary surface clip list */
  LPDIRECTDRAWSURFACE psfcVideo;      /* Live video surface, overlay or offscreen */
  LPDIRECTDRAWSURFACE psfcOverlay;    /* Offscreen graphics overlay */
#endif

  /* Video overlay management */
  BOOL bOverlaySupported;             /* Hardware overlay supported */
    DWORD dwAlignBoundarySrc;
    DWORD dwAlignSizeSrc;
    DWORD dwAlignBoundaryDest;
    DWORD dwAlignSizeDest;
  BOOL bDstOverlayKeySupported;       /* Overlay can be colour keyed onto primary */
  BOOL bSrcBltKeySupported;           /* Blt supports source colour key */
  BOOL bCanBob;                       /* Supports bob deinterlacing */
  BOOL bOverlayStretch;
  BOOL bBltStretch;

  BOOL bStarted;                      /* Video is enabled */
  BOOL bVideoOverlay;                 /* Video surface (psfcVideo) is an overlay */
  BOOL bOverlayVisible;               /* Overlay visibility */
  BOOL bFlipVideo;                    /* Video buffer is a flipping chain */
  ui32 uCurrentBuffer;

  /* Text font */
  enum ASL_Font eFont;
  ui32 fontHeight;
  enum ASL_FontWeight fontWeight;
  ui32 fontStyle;
  COLORREF colorFg, colorBg;
  HFONT hFont;

  HBRUSH hbrBackground;
  };



/* Functions */

/* Get the currently being display scan line */
ASL_EXPORT( ui32, ASL_DisplayGetScanLine( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN:  Display instance */
  ui32*                               /* OUT: Scan line */
));


/* Create a display */
ASL_EXPORT( ui32, ASL_DisplayCreate(  /* Returns !0 on error */
  ASL_DisplayInfo,                    /* IN: Window */
  ui32,                               /* IN: Video pixel width */
  ui32,                               /* IN: Video pixel height */
  ui32,                               /* IN: No. frame buffers, 1..3 */
  ui32,                               /* IN: No. pixel formats */
  ASL_PTR( const ASL_DisplayFormat),  /* IN: Pixel format array */
  ASL_PTR( ASL_DisplayHandle)         /* OUT: instance handle */
));

/* Destroy a display */
ASL_EXPORT( ui32, ASL_DisplayDestroy( /* Returns !0 on error */
  ASL_DisplayHandle                   /* IN Display to destroy */
));

/* Set/get an option flag */
ASL_EXPORT( ui32, ASL_DisplaySetFlag( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ASL_DisplayFlag,                    /* IN: flag */
  ui32                                /* IN: new state */
));
ASL_EXPORT( ui32, ASL_DisplayGetFlag( /* 0= flag not set, 1= flag set, else error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ASL_DisplayFlag                     /* IN: flag */
));

ASL_EXPORT( ui32, ASL_DisplaySetVideoSize( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ui32,                               /* IN: width */
  ui32                                /* IN: height */
));

/* Set/get the video ROI */
ASL_EXPORT( ui32, ASL_DisplaySetVideoRoi( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  i32,                                /* IN: left inclusive */
  i32,                                /* IN: top inclusive */
  i32,                                /* IN: right exclusive */
  i32                                 /* IN: bottom exclusive */
));
ASL_EXPORT( ui32, ASL_DisplayGetVideoRoi( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ASL_PTR( i32),                      /* OUT: left inclusive */
  ASL_PTR( i32),                      /* OUT: top inclusive */
  ASL_PTR( i32),                      /* OUT: right exclusive */
  ASL_PTR( i32)                       /* OUT: bottom exclusive */
));

/* Set/get the display rect */
ASL_EXPORT( ui32, ASL_DisplaySetDisplayRect( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  i32,                                /* IN: left inclusive */
  i32,                                /* IN: top inclusive */
  i32,                                /* IN: right exclusive */
  i32                                 /* IN: bottom exclusive */
));
ASL_EXPORT( ui32, ASL_DisplayGetDisplayRect( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ASL_PTR( i32),                      /* OUT: left inclusive */
  ASL_PTR( i32),                      /* OUT: top inclusive */
  ASL_PTR( i32),                      /* OUT: right exclusive */
  ASL_PTR( i32)                       /* OUT: bottom exclusive */
));

/* Start/stop the display */
ASL_EXPORT( ui32, ASL_DisplayStart(   /* Returns !0 on error */
  ASL_DisplayHandle ,                 /* IN: Display instance */
  ASL_PTR( ASL_DisplayBuffer)         /* OUT: */
));
ASL_EXPORT( ui32, ASL_DisplayStop(    /* Returns !0 on error */
  ASL_DisplayHandle                   /* IN: Display instance */
));

/* Update the display - new video field */
ASL_EXPORT( ui32, ASL_DisplayUpdate(  /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ui32                                /* IN: buffer to display 0.. */
));

/* Get/set the text/graphics overlay key colour */
ASL_EXPORT( ui32, ASL_DisplaySetColourKey( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ui8,                                /* IN: red */
  ui8,                                /* IN: green */
  ui8                                 /* IN: blue */
));
ASL_EXPORT( ui32, ASL_DisplayGetColourKey( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ASL_PTR( ui8),                      /* OUT: red */
  ASL_PTR( ui8),                      /* OUT: green */
  ASL_PTR( ui8)                       /* OUT: blue */
));

#ifdef WIN32
/* Get/release a DC for drawing in the overlay */
ASL_EXPORT( HDC, ASL_DisplayGetDC(    /* Returns NULL on error */
  ASL_DisplayHandle                   /* IN: Display instance */
));
ASL_EXPORT( ui32, ASL_DisplayReleaseDC( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  HDC                                 /* IN */
));
#endif

/* Draw a filled rectangle in the overlay */
ASL_EXPORT( ui32, ASL_DisplayFillRect( /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  i32,                                /* IN: left inclusive */
  i32,                                /* IN: top inclusive */
  i32,                                /* IN: right exclusive */
  i32,                                /* IN: bottom exclusive */
  ui8,                                /* IN: red */
  ui8,                                /* IN: green */
  ui8                                 /* IN: blue */
));

/* text font */
typedef enum ASL_Font
  {
  ASL_kFontFixed = 1,
  ASL_kFontSerif,
  ASL_kFontSansSerif
  } ASL_Font;
enum
  {
  ASL_kFontStyleTop        = (0<<0),
  ASL_kFontStyleBottom     = (1<<0),
  ASL_kFontStyleCentre     = (2<<0),
  ASL_kFontStyleBase       = (3<<0),
  
  ASL_kFontStyleLeft       = (0<<2),
  ASL_kFontStyleRight      = (1<<2),

  ASL_kFontStyleItalic     = (1<<3),
  ASL_kFontStyleUnderline  = (1<<4),
  ASL_kFontStyleTransparent= (1<<5)
  };
typedef enum ASL_FontWeight
  {
  ASL_kFontWeightNormal = 0,
  ASL_kFontWeightLight = 1,
  ASL_kFontWeightBold = 2
  } ASL_FontWeight;
typedef struct ASL_FontInfo
  {
  enum ASL_Font eFont;
  ui32 height;
  enum ASL_FontWeight weight;
  ui32 style;           /* ASL_kFontStyle flags */
  struct { ui8 red, green, blue; } fg;
  struct { ui8 red, green, blue; } bg;
  } ASL_FontInfo;

/* Draw text in the overlay */
ASL_EXPORT( ui32, ASL_DisplayText(    /* Returns !0 on error */
  ASL_DisplayHandle,                  /* IN: Display instance */
  ASL_PTR( const char),               /* IN: ASCIIZ text */
  i32,                                /* IN: left */
  i32,                                /* IN: top */
  ASL_PTR( const ASL_FontInfo)        /* IN: font */
));

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TMG_OVL_H_ */

