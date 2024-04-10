#pragma once
#ifndef _ADL_H_
#define _ADL_H_

#if defined(_MSC_VER) && _MSC_VER >= 1600
#include <stdint.h>
#else
#ifndef _STDINT
typedef char int8_t;
typedef char char8_t;
typedef unsigned char uint8_t;
typedef unsigned char uchar8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#endif // _STDINT
#endif

#define _CALLTYPE_STD __stdcall
#define _CALLTYPE_C __cdecl
#define _CALLTYPE_FAST __fastcall
#define _CALLTYPE_THIS __thiscall

typedef uint32_t err_t;

#define ASL_NO_ERROR 0
#define ASL_INVALID_ENUM 1
#define ASL_OUT_OF_RANGE_VALUE 2
#define ASL_INVALID_OPERATION 3
#define ASL_OUT_OF_MEMORY 4
#define ASL_NOT_SUPPORTED 5
#define ASL_DYNAMIC_CAST_ERROR 6
#define ASL_RUNTIME_ERROR 7

#define ASL_SUCCEEDED(hr) ((hr) == ASL_NO_ERROR)
#define ASL_FAILED(hr) (!ASL_SUCCEEDED(hr))

enum ADL_PIXEL_FORMAT {
    ADL_PIXEL_FORMAT_UNKNOWN = 0,
    ADL_PIXEL_FORMAT_YUV422,
    ADL_PIXEL_FORMAT_Y8,
    ADL_PIXEL_FORMAT_Y16,
    ADL_PIXEL_FORMAT_RGBX32,
    ADL_PIXEL_FORMAT_BGRX32,
    ADL_PIXEL_FORMAT_XBGR32,
    ADL_PIXEL_FORMAT_XRGB32,
    ADL_PIXEL_FORMAT_RGB24,
    ADL_PIXEL_FORMAT_BGR24,
    ADL_PIXEL_FORMAT_BAYER_GBRG,
    ADL_PIXEL_FORMAT_BAYER_BGGR,
    ADL_PIXEL_FORMAT_BAYER_GRBG,
    ADL_PIXEL_FORMAT_BAYER_RGGB
};

enum ADL_IMAGE_ATTRIBUTE {
    ADL_IMAGE_ATTRIBUTE_UNKNOWN = 0,
    ADL_IMAGE_ATTRIBUTE_BITS,
    ADL_IMAGE_ATTRIBUTE_WIDTH,
    ADL_IMAGE_ATTRIBUTE_HEIGHT,
    ADL_IMAGE_ATTRIBUTE_PIXELFORMAT,
    ADL_IMAGE_ATTRIBUTE_STRIDE,
    ADL_IMAGE_ATTRIBUTE_BYTES_PER_LINE,
    ADL_IMAGE_ATTRIBUTE_NUM_BYTES_DATA,
    ADL_IMAGE_ATTRIBUTE_BYTES_PER_PIXEL
};

enum ADL_DISPLAY_ATTRIBUTE {
    ADL_DISPLAY_ATTRIBUTE_UNKNOWN = 0,
    ADL_DISPLAY_ATTRIBUTE_WINDOW,
    ADL_DISPLAY_ATTRIBUTE_STRETCH_TARGETS_WITH_DISPLAY,
    ADL_DISPLAY_ATTRIBUTE_STRETCH_TARGETS_TO_FIT_DISPLAY,
    ADL_DISPLAY_ATTRIBUTE_TARGETS_KEEP_ASPECT_RATIO,
    ADL_DISPLAY_ATTRIBUTE_FULL_SCREEN,
    ADL_DISPLAY_ATTRIBUTE_RECTANGLE,
    ADL_DISPLAY_ATTRIBUTE_VSYNC
};

enum ADL_TARGET_TYPE { ADL_TARGET_TYPE_UNKNOWN = 0, ADL_TARGET_TYPE_RECT };

enum ADL_TARGET_ATTRIBUTE {
    ADL_TARGET_ATTRIBUTE_UNKNOWN = 0,
    ADL_TARGET_ATTRIBUTE_TYPE,
    ADL_TARGET_ATTRIBUTE_IMAGE,
    ADL_TARGET_ATTRIBUTE_ROI,
    ADL_TARGET_ATTRIBUTE_GEOMETRY,
    ADL_TARGET_ATTRIBUTE_LOCATION,
    ADL_TARGET_ATTRIBUTE_STRETCH_WITH_DISPLAY,
    ADL_TARGET_ATTRIBUTE_STRETCH_TO_FIT_DISPLAY,
    ADL_TARGET_ATTRIBUTE_KEEP_ASPECT_RATIO,
    ADL_TARGET_ATTRIBUTE_PROCESSING,
    ADL_TARGET_ATTRIBUTE_PROCESSING_COLORKEY,
    ADL_TARGET_ATTRIBUTE_PROCESSING_CONVOLUTION_WIDTH,
    ADL_TARGET_ATTRIBUTE_PROCESSING_CONVOLUTION_HEIGHT,
    ADL_TARGET_ATTRIBUTE_PROCESSING_CONVOLUTION_COEFICIENTS,
    ADL_TARGET_ATTRIBUTE_END
};

enum ADL_TARGET_PROCESSING_LIST {
    ADL_TARGET_PROCESSING_LIST_NONE = 0,
    ADL_TARGET_PROCESSING_LIST_COLORKEY = 1 << 1,
    ADL_TARGET_PROCESSING_LIST_CONVOLUTION = 1 << 2
};

enum ADL_FONT_ATTRIBUTE {
    ADL_FONT_ATTRIBUTE_UNKNOWN = 0,
    ADL_FONT_ATTRIBUTE_HEIGHT,
    ADL_FONT_ATTRIBUTE_WIDTH,
    ADL_FONT_ATTRIBUTE_WEIGHT,
    ADL_FONT_ATTRIBUTE_ITALIC,
    ADL_FONT_ATTRIBUTE_CHARSET,
    ADL_FONT_ATTRIBUTE_OUTPUTPRECISION,
    ADL_FONT_ATTRIBUTE_QUALITY,
    ADL_FONT_ATTRIBUTE_PITCHANDFAMILY,
    ADL_FONT_ATTRIBUTE_FACENAME
};

typedef struct _ADL_POINT {
    int32_t x;
    int32_t y;
} ADL_POINT;
typedef struct _ADL_RECT {
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
} ADL_RECT;

#define ADL_ARGB_ALPHA(c) (((c) >> 24) & 255)
#define ADL_ARGB_RED(c) (((c) >> 16) & 255)
#define ADL_ARGB_GREEN(c) (((c) >> 8) & 255)
#define ADL_ARGB_BLUE(c) (((c) >> 0) & 255)

#define ADL_ARGB(a, r, g, b)                                                   \
    ((uint32_t)((((a)&0xff) << 24) | (((r)&0xff) << 16) | (((g)&0xff) << 8) |  \
                ((b)&0xff)))

#ifndef WIN64
#define ADL_ATTRIBUTE_TYPE void * // uint32_t
#else
#define ADL_ATTRIBUTE_TYPE void * // ui64
#endif

#define ADL_WCHAR char

// Image
#if defined(__cplusplus) && !defined(CINTERFACE)

typedef struct IADL_Image {
    virtual err_t _CALLTYPE_C Init() = 0;
    virtual err_t _CALLTYPE_C SetAttribute(enum ADL_IMAGE_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
    virtual err_t _CALLTYPE_C GetAttribute(enum ADL_IMAGE_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
} IADL_Image;

#else

typedef struct IADL_Image {
    const struct IADL_ImageVtbl FAR *lpVtbl;
} IADL_Image;

struct IADL_ImageVtbl {
    err_t(_CALLTYPE_C *Init)(IADL_Image FAR *This);
    err_t(_CALLTYPE_C *SetAttribute)(IADL_Image FAR *This,
                                     enum ADL_IMAGE_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
    err_t(_CALLTYPE_C *GetAttribute)(IADL_Image FAR *This,
                                     enum ADL_IMAGE_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
};
#endif

// DisplayFont
#if defined(__cplusplus) && !defined(CINTERFACE)

typedef struct IADL_Font {
    virtual err_t _CALLTYPE_C Init() = 0;
    virtual err_t _CALLTYPE_C SetAttribute(enum ADL_FONT_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
    virtual err_t _CALLTYPE_C GetAttribute(enum ADL_FONT_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
    virtual err_t _CALLTYPE_C Text(const char *pztext, ADL_RECT *pRect,
                                   uint32_t Color) = 0;
    virtual err_t _CALLTYPE_C TextW(const ADL_WCHAR *pztext, ADL_RECT *pRect,
                                    uint32_t Color) = 0;
} IADL_Font;

#else

typedef struct IADL_Font {
    const struct IADL_FontVtbl FAR *lpVtbl;
} IADL_Font;

struct IADL_FontVtbl {
    err_t(_CALLTYPE_C *Init)(IADL_Font FAR *This);
    err_t(_CALLTYPE_C *SetAttribute)(IADL_Font FAR *This,
                                     enum ADL_FONT_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
    err_t(_CALLTYPE_C *GetAttribute)(IADL_Font FAR *This,
                                     enum ADL_FONT_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
    err_t(_CALLTYPE_C *Text)(IADL_Font FAR *This, const char *pztext,
                             ADL_RECT *pRect, uint32_t Color);
    err_t(_CALLTYPE_C *TextW)(IADL_Font FAR *This, const ADL_WCHAR *pztext,
                              ADL_RECT *pRect, uint32_t Color);
};
#endif

// Target
#if defined(__cplusplus) && !defined(CINTERFACE)

typedef struct IADL_Target {
    virtual err_t _CALLTYPE_C Init() = 0;
    virtual err_t _CALLTYPE_C SetAttribute(enum ADL_TARGET_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
    virtual err_t _CALLTYPE_C GetAttribute(enum ADL_TARGET_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
} IADL_Target;

#else

typedef struct IADL_Target {
    const struct IADL_TargetVtbl FAR *lpVtbl;
} IADL_Target;

struct IADL_TargetVtbl {
    err_t(_CALLTYPE_C *Init)(IADL_Target FAR *This);
    err_t(_CALLTYPE_C *SetAttribute)(IADL_Target FAR *This,
                                     enum ADL_TARGET_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
    err_t(_CALLTYPE_C *GetAttribute)(IADL_Target FAR *This,
                                     enum ADL_TARGET_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
};

#endif

// Display
#if defined(__cplusplus) && !defined(CINTERFACE)

typedef struct IADL_Display {
    virtual err_t _CALLTYPE_C Init() = 0;
    virtual err_t _CALLTYPE_C SetAttribute(enum ADL_DISPLAY_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
    virtual err_t _CALLTYPE_C GetAttribute(enum ADL_DISPLAY_ATTRIBUTE Attribute,
                                           ADL_ATTRIBUTE_TYPE Arg) = 0;
    virtual err_t _CALLTYPE_C TargetCreate(IADL_Target **ppTarget) = 0;
    virtual err_t _CALLTYPE_C TargetDestroy(IADL_Target **ppTarget) = 0;
    virtual err_t _CALLTYPE_C FontCreate(IADL_Font **ppFont) = 0;
    virtual err_t _CALLTYPE_C Show() = 0;
} IADL_Display;

#else

typedef struct IADL_Display {
    const struct IADL_DisplayVtbl FAR *lpVtbl;
} IADL_Display;

struct IADL_DisplayVtbl {
    err_t(_CALLTYPE_C *Init)(IADL_Display FAR *This);
    err_t(_CALLTYPE_C *SetAttribute)(IADL_Display FAR *This,
                                     enum ADL_DISPLAY_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
    err_t(_CALLTYPE_C *GetAttribute)(IADL_Display FAR *This,
                                     enum ADL_DISPLAY_ATTRIBUTE Attribute,
                                     ADL_ATTRIBUTE_TYPE Arg);
    err_t(_CALLTYPE_C *TargetCreate)(IADL_Display FAR *This,
                                     IADL_Target **pTarget);
    err_t(_CALLTYPE_C *TargetDestroy)(IADL_Display FAR *This,
                                      IADL_Target **pTarget);
    err_t(_CALLTYPE_C *FontCreate)(IADL_Display FAR *This, IADL_Font **pFont);
    err_t(_CALLTYPE_C *Show)(IADL_Display FAR *This);
};

#endif

#if !defined(__cplusplus) || defined(CINTERFACE)
#define IADL_Display_Init(p) (p)->lpVtbl->Init(p)
#define IADL_Display_SetAttribute(p, a, b) (p)->lpVtbl->SetAttribute(p, a, b)
#define IADL_Display_GetAttribute(p, a, b) (p)->lpVtbl->GetAttribute(p, a, b)
#define IADL_Display_TargetCreate(p, a) (p)->lpVtbl->TargetCreate(p, a)
#define IADL_Display_TargetDestroy(p, a) (p)->lpVtbl->TargetDestroy(p, a)
#define IADL_Display_FontCreate(p, a) (p)->lpVtbl->FontCreate(p, a)
#define IADL_Display_Show(p) (p)->lpVtbl->Show(p)
#else
#define IADL_Display_Init(p) (p)->Init()
#define IADL_Display_SetAttribute(p, a, b) (p)->SetAttribute(a, b)
#define IADL_Display_GetAttribute(p, a, b) (p)->GetAttribute(a, b)
#define IADL_Display_TargetCreate(p, a) (p)->TargetCreate(a)
#define IADL_Display_TargetDestroy(p, a) (p)->TargetDestroy(a)
#define IADL_Display_FontCreate(p, a) (p)->FontCreate(a)
#define IADL_Display_Show(p) (p)->Show()
#endif

#if !defined(__cplusplus) || defined(CINTERFACE)
#define IADL_Target_Init(p) (p)->lpVtbl->Init(p)
#define IADL_Target_SetAttribute(p, a, b) (p)->lpVtbl->SetAttribute(p, a, b)
#define IADL_Target_GetAttribute(p, a, b) (p)->lpVtbl->GetAttribute(p, a, b)
#else
#define IADL_Target_Init(p) (p)->Init()
#define IADL_Target_SetAttribute(p, a, b) (p)->SetAttribute(a, b)
#define IADL_Target_GetAttribute(p, a, b) (p)->GetAttribute(a, b)
#endif

#if !defined(__cplusplus) || defined(CINTERFACE)
#define IADL_Font_Init(p) (p)->lpVtbl->Init(p)
#define IADL_Font_SetAttribute(p, a, b) (p)->lpVtbl->SetAttribute(p, a, b)
#define IADL_Font_GetAttribute(p, a, b) (p)->lpVtbl->GetAttribute(p, a, b)
#define IADL_Font_Text(p, a, b, c) (p)->lpVtbl->Text(p, a, b, c)
#define IADL_Font_TextW(p, a, b, c) (p)->lpVtbl->TextW(p, a, b, c)
#else
#define IADL_Font_Init(p) (p)->Init()
#define IADL_Font_SetAttribute(p, a, b) (p)->SetAttribute(a, b)
#define IADL_Font_GetAttribute(p, a, b) (p)->GetAttribute(a, b)
#define IADL_Font_Text(p, a, b, c) (p)->Text(a, b, c)
#define IADL_Font_TextW(p, a, b, c) (p)->TextW(a, b, c)
#endif

#if !defined(__cplusplus) || defined(CINTERFACE)
#define IADL_Image_Init(p) (p)->lpVtbl->Init(p)
#define IADL_Image_SetAttribute(p, a, b) (p)->lpVtbl->SetAttribute(p, a, b)
#define IADL_Image_GetAttribute(p, a, b) (p)->lpVtbl->GetAttribute(p, a, b)
#else
#define IADL_Image_Init(p) (p)->Init()
#define IADL_Image_SetAttribute(p, a, b) (p)->SetAttribute(a, b)
#define IADL_Image_GetAttribute(p, a, b) (p)->GetAttribute(a, b)
#endif

enum ADL_IMPLEMENTATION {
    ADL_IMPLEMENTATION_D3D9 = 1,
    ADL_IMPLEMENTATION_OPENGL
};

#define ADL_EXPORT __declspec(dllexport) _CALLTYPE_C
#define ADL_IMPORT __declspec(dllimport) _CALLTYPE_C

#ifdef ADL_EXPORTS
#define ADL_EXPORT_FN ADL_EXPORT
#else
#define ADL_EXPORT_FN ADL_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

err_t ADL_EXPORT_FN ADL_DisplayCreate(IADL_Display **ppDisplay,
                                      enum ADL_IMPLEMENTATION Impl);
err_t ADL_EXPORT_FN ADL_DisplayDestroy(IADL_Display **ppDisplay,
                                       enum ADL_IMPLEMENTATION Impl);

err_t ADL_EXPORT_FN ADL_ImageCreate(IADL_Image **ppImage);
err_t ADL_EXPORT_FN ADL_ImageDestroy(IADL_Image **ppImage);

err_t ADL_EXPORT_FN ADL_GetLastError(err_t *pErrorCode, size_t *pSize,
                                     char *pszDescString);

#ifdef __cplusplus
}
#endif

#endif //_ADL_H_