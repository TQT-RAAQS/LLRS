/****************************************************************************
 *
 * Active Silicon
 *
 * File name   : adl_api.h
 * Function    : User API for the Active Display Library (ADL 2)
 * Authors     : Emile Dodin, Jean-Philippe Arnaud
 *
 * Copyright (c) 2017 Active Silicon.
 ****************************************************************************
 * Comments:
 * --------
 * This file is the only include file a user (or higher level library) needs 
 * to include in their application in order to use the ADL 2 library.
 *
 ****************************************************************************
 */

#ifndef _ADL_API_H
#define _ADL_API_H

#if defined _WIN32 || defined _WIN64
#define _ADL_WIN
#endif

#if defined _ADL_WIN

#include <windows.h>
#include <ddraw.h>

#if defined(_MSC_VER) && _MSC_VER >= 1600
/* VS2010 provides stdint.h */
#include <stdint.h>
#else
#ifndef _STDINT
typedef unsigned char uint8_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
#endif /* _STDINT */
#endif /* _MSC_VER >= 1600 */

#define ADL_EXPORT __declspec(dllexport) __cdecl
#define ADL_IMPORT __declspec(dllimport) __cdecl

#endif

extern "C" {

   /* ADL type definitions */
   typedef void * tADLDisplay;   /*! Handle to a display instance */
   typedef void * tADLImage;     /*! Handle to an image instance */
   typedef void * tADLWindow;    /*! Handle to a window */

#define ADL_ENABLE   1
#define ADL_DISABLE  0

   typedef struct {
      int x;
      int y;
   } ADL_Point;

   typedef struct {
      int width;
      int height;
   } ADL_Dimensions;

   /*! ADL graphics library initialization types */
   typedef enum {
      /*! ADL will attempt to use Direct3D but will default to GDI if the hardware does not
       *! support Direct3D.
       *!*/
      ADL_GRAPHICS_AUTO = 0,
      /*! Use Direct3D for rendering. */
      ADL_GRAPHICS_D3D,
      /*! Use GDI for rendering. */
      ADL_GRAPHICS_GDI
   } etADLGraphicsLibrary;

   /*! ADL error types */
   typedef enum {
      ADL_OK = 0,
      ADL_ERROR_GRAPHICS_LIB_NOT_SUPPORTED,
      ADL_ERROR_INVALID_POINTER,
      ADL_ERROR_CREATE_FAILED,
      ADL_ERROR_DESTROY_FAILED,
      ADL_ERROR_BAD_HANDLE,
      ADL_ERROR_INITIALIZATION_FAILED,
      ADL_ERROR_CLOSE_FAILED,
      ADL_ERROR_LIBRARY_NOT_INITIALIZED,
      ADL_ERROR_INVALID_PARAM,
      ADL_ERROR_INVALID_PARAM_VALUE,
      ADL_ERROR_RUNTIME_ERROR
   } etADLStat;

   /*! ADL pixel format types */
   /*! These values are the same as the corresponding PHX Bus Format values */
   typedef enum {
      ADL_PIXEL_FORMAT_UNKNOWN = 0,
      ADL_PIXEL_FORMAT_MONO8 = 0xC0022601,
      ADL_PIXEL_FORMAT_MONO16,
      ADL_PIXEL_FORMAT_MONO32,
      ADL_PIXEL_FORMAT_MONO36,
      ADL_PIXEL_FORMAT_BGR5,
      ADL_PIXEL_FORMAT_BGR565,
      ADL_PIXEL_FORMAT_XBGR8,
      ADL_PIXEL_FORMAT_BGRX8,
      ADL_PIXEL_FORMAT_BGR16,
      ADL_PIXEL_FORMAT_RGB5,
      ADL_PIXEL_FORMAT_RGB565,
      ADL_PIXEL_FORMAT_XRGB8,
      ADL_PIXEL_FORMAT_RGBX8,
      ADL_PIXEL_FORMAT_RGB16,
      ADL_PIXEL_FORMAT_BGR101210,
      ADL_PIXEL_FORMAT_RGB101210,
      ADL_PIXEL_FORMAT_BGR8,
      ADL_PIXEL_FORMAT_RGB8,
      ADL_PIXEL_FORMAT_MONO10,
      ADL_PIXEL_FORMAT_MONO12,
      ADL_PIXEL_FORMAT_MONO14,
      ADL_PIXEL_FORMAT_MONO12P = 0xC002261b,
      ADL_PIXEL_FORMAT_BGR12,
      ADL_PIXEL_FORMAT_RGB12,
      ADL_PIXEL_FORMAT_YUV422_8,
      ADL_PIXEL_FORMAT_MONO10P = 0xC0022621,
      ADL_PIXEL_FORMAT_MONO14P,
      ADL_PIXEL_FORMAT_RGBA8,
      ADL_PIXEL_FORMAT_RGBA10,
      ADL_PIXEL_FORMAT_RGBA12,
      ADL_PIXEL_FORMAT_RGBA14,
      ADL_PIXEL_FORMAT_RGBA16,
      ADL_PIXEL_FORMAT_BAYER_GR8,
      ADL_PIXEL_FORMAT_BAYER_RG8,
      ADL_PIXEL_FORMAT_BAYER_GB8,
      ADL_PIXEL_FORMAT_BAYER_BG8,
      ADL_PIXEL_FORMAT_BAYER_GR10,
      ADL_PIXEL_FORMAT_BAYER_RG10,
      ADL_PIXEL_FORMAT_BAYER_GB10,
      ADL_PIXEL_FORMAT_BAYER_BG10,
      ADL_PIXEL_FORMAT_BAYER_GR12,
      ADL_PIXEL_FORMAT_BAYER_RG12,
      ADL_PIXEL_FORMAT_BAYER_GB12,
      ADL_PIXEL_FORMAT_BAYER_BG12,
      ADL_PIXEL_FORMAT_BAYER_GR14,
      ADL_PIXEL_FORMAT_BAYER_RG14,
      ADL_PIXEL_FORMAT_BAYER_GB14,
      ADL_PIXEL_FORMAT_BAYER_BG14,
      ADL_PIXEL_FORMAT_BAYER_GR16,
      ADL_PIXEL_FORMAT_BAYER_RG16,
      ADL_PIXEL_FORMAT_BAYER_GB16,
      ADL_PIXEL_FORMAT_BAYER_BG16,
      ADL_PIXEL_FORMAT_BGR10,
      ADL_PIXEL_FORMAT_RGB10,
      ADL_PIXEL_FORMAT_BGR14,
      ADL_PIXEL_FORMAT_RGB14
   } etADLPixelFormat;

   /*! ADL display object parameters
    *! The parameters are used with the ADL_DisplayParameterSet() and
    *! ADL_DisplayParameterGet() functions.
    *! The ADL_DISPLAY_INIT_xxx parameters can only be set before the
    *! display object gets initialized.
    */
   typedef enum {
      /*! External window handle (HWND) to be associated with the display object.
       *! If no window is specified, the window is internally managed.
       *! Argument type: void *.
       */
      ADL_DISPLAY_INIT_WINDOW_HANDLE = 1,
      /*! Title to be given to the internal window.
       *! Argument type: const char *.
       */
      ADL_DISPLAY_INIT_WINDOW_TITLE,
      /*! Parent window of the display window.
       *! Argument type: void *.
       */
      ADL_DISPLAY_INIT_WINDOW_PARENT,
      /*! Enable full screen mode.
       *! Possible values:
       *! - ADL_DISABLE: Full screen mode is disabled and the display is windowed.
       *!      Default value.
       *! - ADL_ENABLE: Full screen mode is enabled, the display window has no border
       *!      and takes all available screen space.
       *! Argument type: uin32_t *.
       */
      ADL_DISPLAY_INIT_FULL_SCREEN,
      /*! Enable VSync for rendering.
       *! Possible values:
       *! - ADL_DISABLE: VSync disabled. Display will render images as fast as possible, regardless of VSync.
       *! - ADL_ENABLE: VSync enabled. Display will render images in sync with VSync. Default value.
       *! Argument type: uin32_t *.
       */
      ADL_DISPLAY_INIT_USE_VSYNC,
      /*! Allows associating a user-supplied pointer with the display.
       *! Argument type : void *.
       */
      ADL_DISPLAY_INIT_PARENT_CONTEXT,
      /*! The zoom factor. The same factor is applied horizontally and vertically.
       *! The zoom is implemented by replicating the pixels.
       *! Possible values: 0 = no zoom (default value), 1 = x2, 2 = x4, ...
       *! Argument type: uin32_t *.
       */
      ADL_DISPLAY_XY_ZOOM_FACTOR,
      /*! Enable fit-to-display function, whereby the library resizes the image so that it fits within
       *! the display window.
       *! When enabled, the zoom factor is ignored.
       *! Related features: ADL_DISPLAY_KEEP_ASPECT_RATIO and ADL_DISPLAY_SAMPLING_METHOD.
       *! Possible values are:
       *! - ADL_DISABLE: Disabled. Default value.
       *! - ADL_ENABLE: Enabled.
       *! Argument type: uin32_t *.
       */
      ADL_DISPLAY_FIT_TO_DISPLAY,
      /*! Specify a point in the image that will remain at the same position in the display
       *! when the image is zoomed.
       *! Argument type: ADL_Point *.
       */
      ADL_DISPLAY_XY_ZOOM_OFFSET,
      /*! Pixel format used by the display.
       *! Argument type: uin32_t *.
       *! Read-only.
       */
      ADL_DISPLAY_FORMAT,
      /*! Offset to apply to the image AOI currently displayed. Offset used for the next rendering
       *! operation.
       *! Argument type: ADL_Dimensions *.
       */
      ADL_DISPLAY_AOI_OFFSET,
      /*! Return the origin (top-left point) of the image AOI currently displayed.
       *! Argument type: ADL_Point *.
       *! Read-only.
       */
      ADL_DISPLAY_AOI,
      /*! Return the zoom factor used by ADL when fit to display is enabled.
      *! Only valid when ADL_DISPLAY_FIT_TO_DISPLAY = 1 and ADL_DISPLAY_SAMPLING_METHOD = 0.
      *! Argument type: uin32_t *.
      *! Read-only.
      */
      ADL_DISPLAY_FIT_TO_DISPLAY_XY_ZOOM_FACTOR,
      /*! Allow synchronizing the scrollbars of another display with this one.
      *! When enabled, moving the scrollbars of one display also moves the scrollbars of the other.
      *! The argument is the handle of the display to be synchronized.
      *! Argument type: tADLDisplay *.
      */
      ADL_DISPLAY_SYNCED,
      /*! Control if the image aspect ratio should be preserved when the image is over- or sub-sampled
      *! when fit-to-display is enabled.
      *! Only used when ADL_DISPLAY_FIT_TO_DISPLAY = 1 and ADL_DISPLAY_SAMPLING_METHOD = 1; no effect otherwise.
      *! Possible values:
      *! - ADL_DISABLE: Do not preserve aspect ratio.
      *! - ADL_ENABLE: Preserve aspect ratio. Default value.
      *! Argument type: uin32_t *.
      */
      ADL_DISPLAY_KEEP_ASPECT_RATIO,
      /*! The pixel sampling method used when the image is over- or sub-sampled when fit-to-display is enabled.
      *! Only used when ADL_DISPLAY_FIT_TO_DISPLAY = 1; no effect otherwise.
      *! Possible values are:
      *! - 0: Default value.
      *!      The same sampling factor is used horizontally and vertically.
      *!      The image resolution may be reduced or increased by integral factors only (-3, -2, 2, 3, 4, etc.).
      *!      The sampling either drops or duplicates pixels. No interpolation is performed.
      *!      Depending on the combination of image and display resolution, this may result in parts of the
      *!      display being not filled with image data.
      *! - 1: Non-integral image sampling; the sampling factor is calculated so that the image exactly fits the display.
      *!      The resulting sampling factor is non-integral.
      *!      Note, when this option is enabled and the Direct3D renderer is used, the image has to fit within a Direct3D
      *!      texture. The maximum resolution of such a texture can be queried via ADL_DISPLAY_D3D_MAX_TEXTURE_RES.
      *! Argument type: uin32_t *.
      */
      ADL_DISPLAY_SAMPLING_METHOD,
      /*! Return the maximum width and height of a Direct3D texture for the current graphics adapter. Library must have been
      *! initialized with ADL_GRAPHICS_D3D, otherwise an error is generated.
      *! Argument type: ADL_Dimensions *.
      *! Read-only.
      *!*/
      ADL_DISPLAY_D3D_MAX_TEXTURE_RES
   } etADLDisplayParam;

   /* ADL image object parameters
    * ---------------------------
    * The parameters are used with the ADL_ImageParameterSet() and
    * ADL_ImageParameterGet() functions.
    */
   typedef enum {
      /*! Pointer to the buffer containing the image. Note, the image data must be consecutive in the buffer,
      *! line padding is not supported.
      *! Argument type: void *.
      */
      ADL_IMAGE_ADDRESS = 1,
      /*! Image width.
      *! Argument type: uin32_t *.
      */
      ADL_IMAGE_WIDTH,
      /*! Image Height.
      *! Argument type: uin32_t *.
      */
      ADL_IMAGE_HEIGHT,
      /*! Image pixel format.
      *! Argument type: etADLPixelFormat *.
      */
      ADL_IMAGE_PIXEL_FORMAT
   } etADLImageParam;

   /*!
    * \brief Initialize the library, including the graphics library.
    * To be called once, prior to any function call to allow global initialization of the library.
    * \param eGraphicsLibrary The graphics library to use for the rendering.
    * Possible values: See etADLGraphicsLibrary declaration.
    */
   etADLStat ADL_EXPORT ADL_InitLib(etADLGraphicsLibrary eGraphicsLibrary);

   /*!
    * \brief Must be called after no function of the library is needed anymore to free
    * the resources from the ADL_InitLib() function call.
    * Releases and cleans-up the graphics context.
    */
   etADLStat ADL_EXPORT ADL_CloseLib();

   /*!
    * \brief Query what graphics library is used for the rendering.
    * \param eGraphicsLibrary Contains the underlying rendering library used by ADL.
    * \return ADL_ERROR_LIBRARY_NOT_INITIALIZED if the library has not been initialised yet with ADL_InitLib().
    *         ADL_ERROR_INVALID_POINTER if a null pointer is passed for \eGraphicsLibrary.
    *         ADL_OK otherwise.
    */
   etADLStat ADL_EXPORT ADL_LibInfo(etADLGraphicsLibrary *eGraphicsLibrary);

   /*!
    * \brief Create an ADL display object and return the handle to it.
    * ADL_DisplayInit() must be called to initialize the object prior to any other function.
    */
   etADLStat ADL_EXPORT ADL_DisplayCreate(tADLDisplay *phDisplay);

   /*!
    *! \brief Initialize the ADL display object created with ADL_DisplayCreate().
    */
   etADLStat ADL_EXPORT ADL_DisplayInit(tADLDisplay hDisplay);

   /*!
   \brief Destroy an ADL display object.
   */
   etADLStat ADL_EXPORT ADL_DisplayDestroy(tADLDisplay *phDisplay);

   /*!
    * \brief Set parameter in display object.
    * \param hDisplay Handle to the display.
    * \param eDisplayParam The parameter to set.
    * \param pvParamValue Pointer to a variable holding the value for the parameter.
    * See each parameter's documentation for the variable's type to use.
    */
   etADLStat ADL_EXPORT ADL_DisplayParameterSet(tADLDisplay hDisplay, etADLDisplayParam eDisplayParam, void *pvParamValue);

   /*!
    * \brief Retrieve the value of a parameter from the display.
    * \param hDisplay Handle to the display.
    * \param eDisplayParam The parameter to read.
    * \param pvParamValue Pointer to the variable that will receive the value of the parameter.
    * See each parameter's documentation for the variable's type to use.
    */
   etADLStat ADL_EXPORT ADL_DisplayParameterGet(tADLDisplay hDisplay, etADLDisplayParam eDisplayParam, void *pvParamValue);

   /*!
    * \brief Select a new ADL image to be displayed.
    * The image will be processed as soon as the display has completed rendering of the current image.
    * The display does not copy the image internally, so it must remain valid until the following call
    * to ADL_DisplayImageSet().
    * \param hDisplay Handle to the display.
    * \param hImage The new image to display.
    * \param phPreviousImage The previous image that was selected and that is no longer used by the display.
    */
   etADLStat ADL_EXPORT ADL_DisplayImageSet(tADLDisplay hDisplay, tADLImage hImage, tADLImage *phPreviousImage);


   /*!
    * \brief Try to lock the display. Returns immediately. On successful lock acquisition, the display
    * no longer updates and nor uses the current tADLImage.
    * \param hDisplay Handle to the display.
    * \param Locked Null if display not locked, non-null otherwise.
    * \return etADLStat ADL_OK if no error.
    */
   etADLStat ADL_EXPORT ADL_TryLock(tADLDisplay hDisplay, int *Locked);

   /*!
   * \brief Unlock the display. The display must be locked by the current thread of execution (via ADL_TryLock()),
   * otherwise, the behavior is undefined.
   * \param hDisplay Handle to the display.
   * \param Unlocked Null if display not unlocked, non-null otherwise.
   * \return etADLStat ADL_OK if no error.
   */
   etADLStat ADL_EXPORT ADL_Unlock(tADLDisplay hDisplay, int *Unlocked);

   /*
    * \brief Create an ADL image object. The tADLImage object is a self-describing container for an image.
    * It should be initialized using the ADL_ImageParameterSet().
    * \param phImage Pointer to the created image instance.
    */
   etADLStat ADL_EXPORT ADL_ImageCreate(tADLImage *phImage);

   /*!
    * \brief Destroy an ADL image object.
    * \param phImage Pointer to the image instance to be destroyed.
    */
   etADLStat ADL_EXPORT ADL_ImageDestroy(tADLImage *phImage);

   /*!
    * \brief Set a parameter in an image instance.
    * \param hImage Handle to the image.
    * \param eImageParam The parameter to set.
    * \param pvParamValue Pointer to a variable holding the value of the parameter.
    * See each parameter's documentation for the variable's type to use.
    */
   etADLStat ADL_EXPORT ADL_ImageParameterSet(tADLImage hImage, etADLImageParam eImageParam, void *pvParamValue);

   /*!
   * \brief Retrieve the value of a parameter from an image instance.
   * \param hImage Handle to the image.
   * \param eImageParam The parameter to set.
   * \param pvParamValue Pointer to a variable that will receive the value of the parameter.
   * See each parameter's documentation for the variable's type to use.
   */
   etADLStat ADL_EXPORT ADL_ImageParameterGet(tADLImage hImage, etADLImageParam eImageParam, void *pvParamValue);

   /*!
    * \brief Retrieve the last error from the library.
    * \param peStat Pointer to the last error code.
    * \param pszErrorMessage User-allocated array of characters that will contain a copy of the last error message.
    * \param pnSize [in] Pointer to the size in bytes of pszErrorMessage.
    [out] Pointer to the actual size in bytes of the error message copied to pszErrorMessage.
    */
   etADLStat ADL_EXPORT ADL_GetLastError(etADLStat *peStat, char *pszErrorMessage, size_t *pnSize);
}

#endif /* #define _ADL_API_H */
