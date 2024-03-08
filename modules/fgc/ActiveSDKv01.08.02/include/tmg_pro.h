/****************************************************************************
 *
 * ACTIVE SILICON LIMITED
 *
 * File name   : tmg_pro.h
 * Function    : Prototype definitions - include file.
 * Project     : TMG
 * Authors     : Colin Pearce
 * Systems     : ANSI C, Various.
 * Version     : See tmg_api.h for overall library release information.
 * Release Date: See tmg_api.h for overall library release information.
 *
 * Copyright (c) 1993-2001 Active Silicon Ltd.
 ****************************************************************************
 *
 * File History
 * ------------
 *
 * Comments:
 * --------
 * "non-doc" means undocumented function that may be used occasionally in
 * applications.
 * "non-doc/user" means undocumented and not intended to be used outside the
 * libraries.
 * Note all the functions required by Snapper libraries are in tmg_snp.c, but
 * the functions are still grouped logically in this header file.
 *
 * File History
 * ------------
 * 11-Mar-97, HCP, "#ifndef Thandle" section added so that the file will
 *                 compile independently (of Snapper library).
 * 
 * <Top level history information goes in tmg_api.h>
 *
 */

#ifndef _TMG_PRO_H_
#define _TMG_PRO_H_


/*
 * Prototype Definitions
 * =====================
 */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * TMG_cnv1.c - TMG image conversion routines - non-YUV422
 * ----------
TMG_cnv1.c(ForBrief)
 */
Terr EXPORT_FN TMG_image_convert(Thandle, Thandle, ui16, ui32, ui16);
Terr EXPORT_FN TMG_convert_to_paletted_LUT(Thandle, Thandle, ui16); /* non-doc */
Terr EXPORT_FN TMG_convert_to_RGB24(Thandle, Thandle, ui16);        /* non-doc */       /* Thread/alignment safe */
Terr TMG_convert_to_RGB24_1_strip(Thandle, Thandle);                /* non-doc/user */
Terr EXPORT_FN TMG_convert_to_RGBX32(Thandle, Thandle, ui16);       /* non-doc */
Terr EXPORT_FN TMG_convert_to_BGRX32(Thandle, Thandle, ui16);       /* non-doc */       /* Thread/alignment safe */
Terr EXPORT_FN TMG_convert_to_XBGR32(Thandle, Thandle, ui16);       /* non-doc */
Terr EXPORT_FN TMG_convert_to_XRGB32(Thandle, Thandle, ui16);       /* non-doc */
Terr EXPORT_FN TMG_convert_to_BGR24(Thandle, Thandle, ui16);        /* non-doc */       /* Thread/alignment safe */
Terr EXPORT_FN TMG_convert_to_RGB16(Thandle, Thandle, ui16);        /* non-doc */
Terr EXPORT_FN TMG_convert_to_RGB15(Thandle, Thandle, ui16);        /* non-doc */
Terr TMG_convert_to_RGB15_1_strip(Thandle, Thandle);                /* non-doc/user */
Terr EXPORT_FN TMG_convert_to_RGB8(Thandle, Thandle, ui16);         /* non-doc */
Terr EXPORT_FN TMG_convert_to_RGB8_dither(Thandle, Thandle, ui16);  /* non-doc */
Terr EXPORT_FN TMG_paletted_to_RGB24_or_Y8(Thandle, Thandle, ui16); /* non-doc */       
Terr EXPORT_FN TMG_convert_to_CMYK32(Thandle, Thandle, ui16);       /* non-doc */ 
Terr EXPORT_FN TMG_convert_to_Y8(Thandle, Thandle, ui16);           /* non-doc */       /* Thread/alignment safe */
Terr EXPORT_FN TMG_convert_to_Y16(Thandle, Thandle, ui16);          /* non-doc */       /* Thread/alignment safe */

/*
 * TMG_cnv2.c - TMG image conversion routines - YUV422
 * ----------
TMG_cnv2.c(ForBrief)
 */
Terr EXPORT_FN TMG_YUV422_to_HSI(Timage_handle Hin_image, Timage_handle Hout_image, ui16 TMG_action); /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_RGBX32(Thandle, Thandle, ui16);       /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_BGRX32(Thandle, Thandle, ui16);       /* non-doc */     /* Thread/alignment safe */
Terr EXPORT_FN TMG_YUV422_to_XBGR32(Thandle, Thandle, ui16);       /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_XRGB32(Timage_handle Hin_image, Timage_handle Hout_image, ui16 TMG_action);
Terr EXPORT_FN TMG_YUV422_to_RGB24(Thandle, Thandle, ui16);        /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_BGR24(Thandle, Thandle, ui16);        /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_RGB16(Thandle, Thandle, ui16);        /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_RGB15(Thandle, Thandle, ui16);        /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_paletted_LUT(Thandle, Thandle, ui16); /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_RGBX32_LUT(Thandle, Thandle, ui16);   /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_BGRX32_LUT(Thandle, Thandle, ui16);   /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_XBGR32_LUT(Thandle, Thandle, ui16);   /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_XRGB32_LUT(Timage_handle Hin_image, Timage_handle Hout_image, ui16 TMG_action);
Terr EXPORT_FN TMG_YUV422_to_RGB24_LUT(Thandle, Thandle, ui16);    /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_BGR24_LUT(Thandle, Thandle, ui16);    /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_RGB16_LUT(Thandle, Thandle, ui16);    /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_RGB15_LUT(Thandle, Thandle, ui16);    /* non-doc */
Terr EXPORT_FN TMG_YUV422_to_Y8(Thandle, Thandle, ui16);           /* non-doc */     /* Thread/alignment safe */
Terr EXPORT_FN TMG_YUV422_to_Y16(Thandle, Thandle, ui16);           /* non-doc */
Terr EXPORT_FN TMG_convert_to_YUV422(Thandle, Thandle, ui16);       /* non-doc */

/*
 * TMG_cnv3.c - TMG image conversion routines - LUT functions.
 * ----------
TMG_cnv3.c(ForBrief)
 */
Terr EXPORT_FN TMG_image_conv_LUT_generate(Thandle, ui16);
Terr EXPORT_FN TMG_image_conv_LUT_destroy(ui16);
Terr EXPORT_FN TMG_image_conv_LUT_load(Thandle, ui16, char*);
Terr EXPORT_FN TMG_image_conv_LUT_save(Thandle, ui16, char*);
Terr EXPORT_FN TMG_generate_RGB16_to_paletted_LUT(Thandle);  /* non-doc */
Terr EXPORT_FN TMG_generate_Y16_to_paletted_LUT(Thandle);    /* non-doc */
Terr EXPORT_FN TMG_generate_Y8_to_paletted_LUT(Thandle);     /* non-doc */
Terr EXPORT_FN TMG_destroy_RGB16_to_paletted_LUT(void);      /* non-doc */
Terr EXPORT_FN TMG_destroy_Y16_to_paletted_LUT(void);        /* non-doc */
Terr EXPORT_FN TMG_destroy_Y8_to_paletted_LUT(void);         /* non-doc */
Terr EXPORT_FN TMG_generate_YUV422_to_RGB16_LUT(void);       /* non-doc */
Terr EXPORT_FN TMG_generate_YUV422_to_RGB15_LUT(void);       /* non-doc */
Terr EXPORT_FN TMG_generate_YUV422_to_paletted_LUT(Thandle); /* non-doc */
Terr EXPORT_FN TMG_destroy_YUV422_to_RGB16_LUT(void);        /* non-doc */
Terr EXPORT_FN TMG_destroy_YUV422_to_RGB15_LUT(void);        /* non-doc */
Terr EXPORT_FN TMG_destroy_YUV422_to_paletted_LUT(void);     /* non-doc */

/*
 * tmg_crop.c - see also tmg_spl.c
 * ----------
tmg_crop.c(ForBrief)
 */
Terr EXPORT_FN TMG_IP_crop(Thandle, Thandle, i32[], ui16);                                        /* Thread/alignment safe */
Terr EXPORT_FN TMG_IP_image_insert(Thandle hInImage, Thandle hInsertImage, ui32 dwX, ui32 dwY);

/*
 * TMG_dib.c - Windows DIBs/BMPs
 * ---------
TMG_dib.c(ForBrief)
 */
Terr EXPORT_FN TMG_convert_to_DIB_w31(Thandle, Thandle, ui32, ui16);   /* non-doc */
Terr EXPORT_FN TMG_read_BMP_file(Timage_handle, ui16);                 /* non-doc */
Terr EXPORT_FN TMG_write_BMP_file(Timage_handle, ui16);                /* non-doc */
Terr EXPORT_FN TMG_image_write_BMP_buffer(Timage_handle Himage, ui8 *pbBuffer, ui32 *pdwByteCount, ui16 TMG_action);

/*
 * TMG_disp.c - TMG display routines
 * ----------
TMG_disp.c(ForBrief)
 */
/* Generic to all operating systems */
/* -------------------------------- */
Terr     EXPORT_FN TMG_display_create(void);                          /* Thread/alignment safe */
Terr     EXPORT_FN TMG_display_destroy(Thandle);                      /* Thread/alignment safe */
Tboolean EXPORT_FN TMG_display_get_flags(Thandle, ui32);
Terr     EXPORT_FN TMG_display_get_parameter(Thandle, ui16);
Terr     EXPORT_FN TMG_display_get_ROI(Thandle, i16*);
Terr     EXPORT_FN TMG_display_image(Tdisplay_handle Hdisplay, Thandle Hin_image, ui16 TMG_action);
Terr     EXPORT_FN TMG_display_set_flags(Thandle, ui32, Tboolean);
Terr     EXPORT_FN TMG_display_set_parameter(Thandle, ui16, ui32);
Terr     EXPORT_FN TMG_display_set_ROI(Tdisplay_handle, Tparam, i16*);

/*
 * tmg_dos.c
 * ---------
tmg_dos.c(ForBrief)
 */
Terr EXPORT_FN TMG_DRV_display_image(Tdisplay_handle, Thandle, ui16);

#if defined _TMG_FG_GRAPHICS || defined _TMG_GRX_GRAPHICS
Terr EXPORT_FN TMG_display_init(Tdisplay_handle, ui16);
Terr EXPORT_FN TMG_display_clear(Tdisplay_handle);
Terr EXPORT_FN TMG_display_box_fill(Tdisplay_handle, ui16, i16*);
Terr EXPORT_FN TMG_display_draw_text(Tdisplay_handle, char*, ui16, ui16);
Terr EXPORT_FN TMG_display_cmap_install(Tdisplay_handle, Thandle);
Terr EXPORT_FN TMG_display_cmap(Tdisplay_handle, Timage_handle, ui16);
Terr EXPORT_FN TMG_display_set_font(Tdisplay_handle, ui16);
#endif

/*
 * tmg_x.c
 * ---------
tmg_x.c(ForBrief)
 */
#ifdef _TMG_X_GRAPHICS
Terr EXPORT_FN TMG_display_init(Thandle, ui16);
Terr EXPORT_FN TMG_display_cmap_install(Thandle, Thandle);
Terr EXPORT_FN TMG_display_clear(Thandle);
Terr EXPORT_FN TMG_DRV_display_image(Thandle, Thandle, ui16);
Terr EXPORT_FN TMG_display_set_Xid(Thandle, ui32, Window);
#endif

#if defined _TMG_WINDOWS
/* tmg_w32.c
 * ---------
 */
/* User functions: */
Terr EXPORT_FN TMG_display_init(Tdisplay_handle, HWND);            /* Thread/alignment safe */
Terr EXPORT_FN TMG_display_set_hWnd(Tdisplay_handle, HWND);
Terr EXPORT_FN TMG_display_set_paint_hDC(Tdisplay_handle, HDC);
HWND EXPORT_FN TMG_display_get_hWnd(Tdisplay_handle);
HDC  EXPORT_FN TMG_display_get_paint_hDC(Tdisplay_handle);
Terr EXPORT_FN TMG_DRV_display_image(Tdisplay_handle, Thandle, ui16);
Terr EXPORT_FN TMG_display_print_DIB(Tdisplay_handle, Thandle, i16, ui16);

/* Non-user functions: */
Terr EXPORT_FN TMG_display_DIB(Tdisplay_handle, Thandle, ui16);  /* non-doc */
Terr EXPORT_FN TMG_display_DDB(Tdisplay_handle, Thandle, ui16);  /* non-doc */

/* tmg_dd.c
 * --------
 * Display function that uses Direct Draw under Windows 95 and Windows NT.
tmg_dd.c(ForBrief)
 */
Terr EXPORT_FN TMG_display_direct_draw_w95(Thandle, Thandle, ui16 ); /* non-doc */

/* tmg_ovl.c
 * ---------
 */
Terr EXPORT_FN TMG_display_GetScanLine( ui32 hDisplay, ui32 *pnScanLine);
Terr EXPORT_FN TMG_display_OverlayCreate( Tdisplay_handle, const ui32[], ui32);
Terr EXPORT_FN TMG_display_OverlayDestroy( Tdisplay_handle);
Terr EXPORT_FN TMG_display_OverlayStart( Tdisplay_handle, ui32*, void**, ui32*);
Terr EXPORT_FN TMG_display_OverlayStop( Tdisplay_handle);
Terr EXPORT_FN TMG_display_OverlaySetDisplayRegion( ui32 hDisplay, i16 Roi[] );
void EXPORT_FN TMG_display_OverlayUpdate( Tdisplay_handle);

ASL_DisplayHandle EXPORT_FN TMG_display_OverlayGetHandle( Tdisplay_handle);
HDC EXPORT_FN     TMG_display_Overlay_Get_hDC( Tdisplay_handle hDisplay);
Terr EXPORT_FN    TMG_display_Overlay_Release_hDC( ui32 hDisplay, HDC hDC);
Terr EXPORT_FN    TMG_display_OverlayBitBlt( ui32 hDisplay, void *pvBitmapData, i16 RoiImage[]);
Terr EXPORT_FN    TMG_display_OverlayFillRegion( ui32 hDisplay, i16 RoiRect[], ui8 bRed, ui8 bGrn, ui8 bBlu);

#endif  /* _TMG_WINDOWS */ 


/*
 * tmg_mac.c
 * ---------
tmg_mac.c(ForBrief)
 */
#if defined _TMG_MAC  /* MacOS display functions */
Terr EXPORT_FN TMG_display_init(Tdisplay_handle, PixMapHandle);
Terr EXPORT_FN TMG_DRV_display_image(Tdisplay_handle, Thandle, ui16);
Terr EXPORT_FN TMG_display_set_mask(Tdisplay_handle, RgnHandle);
#endif  /* MacOS specific display functions */

#if defined _TMG_MACOSX && !defined __MWERKS__
Terr EXPORT_FN TMG_display_init(Thandle, ui16);
Terr EXPORT_FN TMG_display_cmap_install(Thandle, Thandle);
Terr EXPORT_FN TMG_display_clear(Thandle);
Terr EXPORT_FN TMG_DRV_display_image(Thandle, Thandle, ui16);
Terr EXPORT_FN TMG_display_set_context(Thandle, CGLContextObj);
#endif


/*
 * TMG_eps.c - TMG encapsulated PostScript routines.
 * ---------
TMG_eps.c(ForBrief)
 */
Terr EXPORT_FN TMG_write_EPS_file(Thandle, ui16);
Terr EXPORT_FN TMG_read_EPS_file(Thandle, ui16);
char EXPORT_FN_PTR TMG_sscanf(char *Pbuf, ui16 *Pnumber);

/* 
 * TMG_gen.c - General Image Processing Routines.
 * ---------
TMG_gen.c(ForBrief)
 * Some of these files are in tmg_snp.c.
 */
void EXPORT_FN TMG_ErrHandlerDefault(ui32 hCard, char *pszFnName, ui32 dwErrCode, char *pszDescString);
#if defined _TMG_WINDOWS || defined _TMG_MAC
ui32 EXPORT_FN TMG_ErrHandlerInstall(void (*pFnHandler)(ui32, char*, ui32, char*));
#else /* _TMG_DOS32 & others */
ui32 EXPORT_FN TMG_ErrHandlerInstall(void (EXPORT_FN *pFnHandler)(ui32, char*, ui32, char*));
#endif
void EXPORT_FN TMG_ErrProcess(ui32 hLib, char *pszFnName, ui32 dwErrCode, char *pszDescString);

Terr     EXPORT_FN     TMG_cmap_copy(Thandle, Thandle);
void     EXPORT_FN_PTR TMG_DebugMalloc( int nNumBytes);
void     EXPORT_FN     TMG_DebugFree( void *pData);
void     EXPORT_FN     TMG_DebugMallocPrint( int nRef);
Terr     EXPORT_FN     TMG_image_calc_skip_bytes(ui32, ui32, ui32);
Terr     EXPORT_FN     TMG_image_calc_total_strips(Thandle);
Terr     EXPORT_FN     TMG_image_check(Thandle);
ui32     EXPORT_FN     TMG_ImageCheck2(void);
Terr     EXPORT_FN     TMG_image_copy(Thandle, Thandle);                  /* Thread/alignment safe */
Terr     EXPORT_FN     TMG_image_copy_parameters(Thandle, Thandle);       /* Thread/alignment safe */
Terr     EXPORT_FN     TMG_image_copy_parameters_raw(Thandle, Thandle);   /* Thread/alignment safe */
Thandle  EXPORT_FN     TMG_image_create(void);                            /* Thread/alignment safe */
Terr     EXPORT_FN     TMG_image_destroy(Thandle);                        /* Thread/alignment safe */
Terr     EXPORT_FN     TMG_ImageUnlockAndDestroy( Thandle hImage);
Terr     EXPORT_FN     TMG_image_free_data(Thandle);                      /* Thread/alignment safe */
ui32     EXPORT_FN     TMG_image_get_parameter(Thandle, ui16);            /* Thread/alignment safe */
void     EXPORT_FN_PTR TMG_image_get_ptr(Thandle, ui16);                  /* Thread/alignment safe */
Terr     EXPORT_FN     TMG_image_get_flags(Thandle, ui32);                /* Thread/alignment safe */
char     EXPORT_FN_PTR TMG_image_get_infilename(Thandle);                 /* Thread/alignment safe */
char     EXPORT_FN_PTR TMG_image_get_outfilename(Thandle);                /* Thread/alignment safe */
Terr     EXPORT_FN     TMG_image_malloc_a_strip(Thandle);                 /* Thread/alignment safe */
ui32     EXPORT_FN     TMG_ImageMake( ui32 hImage, i32 nWidth, i32 nHeight, ui16 wPixelFormat, ui32 dwColour);
Terr     EXPORT_FN     TMG_image_move(Thandle, Thandle);
Terr     EXPORT_FN     TMG_image_set_flags(Thandle, ui32, Tboolean);
Terr     EXPORT_FN     TMG_image_set_infilename(Thandle, char*);
Terr     EXPORT_FN     TMG_image_set_outfilename(Thandle, char*);
Terr     EXPORT_FN     TMG_image_set_parameter(Thandle, ui16, ui32);
Terr     EXPORT_FN     TMG_image_set_ptr(Thandle, ui16, void*);
Terr     EXPORT_FN     TMG_InitLibrary(void);
Terr     EXPORT_FN     TMG_LibraryInit(void);
Terr     EXPORT_FN     TMG_Library_Init(void);
Terr     EXPORT_FN     TMG_Init(void);

/* One line functions - we'll keep for now */
Terr     EXPORT_FN     TMG_get_width(Thandle);            /* non-user/doc */
Terr     EXPORT_FN     TMG_get_height(Thandle);           /* non-user/doc */
Terr     EXPORT_FN     TMG_get_depth(Thandle);            /* non-user/doc */
Terr     EXPORT_FN     TMG_get_lines_this_strip(Thandle); /* non-user/doc */
Terr     EXPORT_FN     TMG_get_bytes_per_line(Thandle);   /* non-user/doc */


/* Internal non-user functions */
#if defined _TMG_MAC
EXPORT_FN struct Tcmap* TMG_image_create_cmap(void);      /* non-user/doc */
#else
struct Tcmap EXPORT_FN_PTR TMG_image_create_cmap(void);   /* non-user/doc */
#endif
Terr     EXPORT_FN     TMG_copy_outfilename(Thandle, Thandle); /* non-user/doc */
IM_UI8   EXPORT_FN_PTR TMG_malloc_N_lines(Thandle, ui16);      /* non-user/doc */

ui16 EXPORT_FN TMG_fget_ui16i(FILE *Pfile);
ui32 EXPORT_FN TMG_fget_ui32m(FILE *Pfile);
ui32 EXPORT_FN TMG_fget_ui32i(FILE *Pfile);
ui16 EXPORT_FN TMG_fget_ui16m(FILE *Pfile);
void EXPORT_FN TMG_fput_ui32m(FILE *Pfile, ui32 b);
void EXPORT_FN TMG_fput_ui16m(FILE *Pfile, ui16 b);
void EXPORT_FN TMG_fput_ui8(FILE *Pfile, ui8 b);
void EXPORT_FN TMG_fput_ui16i(FILE *Pfile, ui16 b);
void EXPORT_FN TMG_fput_ui32i(FILE *Pfile, ui32 b);

ui16 EXPORT_FN TMG_mget_ui16i(ui8 *pbBuf);
ui32 EXPORT_FN TMG_mget_ui32m(ui8 *pbBuf);
ui32 EXPORT_FN TMG_mget_ui32i(ui8 *pbBuf);
ui16 EXPORT_FN TMG_mget_ui16m(ui8 *pbBuf);
void EXPORT_FN TMG_mput_ui32m(ui8 *pbBuf, ui32 b);
void EXPORT_FN TMG_mput_ui16m(ui8 *pbBuf, ui16 b);
void EXPORT_FN TMG_mput_ui8(ui8 *pbBuf, ui8 b);
void EXPORT_FN TMG_mput_ui16i(ui8 *pbBuf, ui16 b);
void EXPORT_FN TMG_mput_ui32i(ui8 *pbBuf, ui32 b);

ui32 EXPORT_FN TMG_Util_ParseCommand(struct tTMG_Cmd *psCmd, int argc, char **argv);


/*
 * TMG_jpg.c
 * ---------
TMG_jpg.c(ForBrief)
 */
Terr     EXPORT_FN     TMG_JPEG_build_image(Thandle, Thandle, ui16);
Terr     EXPORT_FN     TMG_JPEG_copy_parameters(Thandle, Thandle); /* non-doc/user */
Terr     EXPORT_FN     TMG_JPEG_destroy_Tjpeg(Thandle);            /* non-doc/user */
Terr     EXPORT_FN     TMG_JPEG_file_close(Thandle);
Terr     EXPORT_FN     TMG_JPEG_file_open(Thandle);
Terr     EXPORT_FN     TMG_JPEG_file_read(Thandle);
Terr     EXPORT_FN     TMG_JPEG_buffer_read(Thandle Himage, ui8 *pbData, ui32 dwBytesData);
Terr     EXPORT_FN     TMG_JPEG_file_write(Thandle, ui16);
Terr     EXPORT_FN     TMG_JPEG_buffer_write(Timage_handle Hjpeg_image, ui8 *pdData, ui32 *pdwCount, ui16 TMG_action);
Terr     EXPORT_FN     TMG_JPEG_free_data(Thandle);            /* non-doc/user */
FILE     EXPORT_FN_PTR TMG_JPEG_get_file_ptr(Thandle);         /* non-doc/user */
Terr     EXPORT_FN     TMG_JPEG_get_parameter(Thandle, ui16);  /* non-doc/user */
Thandle  EXPORT_FN     TMG_JPEG_image_create(void);
Terr     EXPORT_FN     TMG_JPEG_malloc_data(Thandle);               /* non-doc/user */
Terr     EXPORT_FN     TMG_JPEG_malloc_N_bytes_data(Thandle, ui32); /* non-doc/user */
Terr     EXPORT_FN     TMG_JPEG_sequence_build(Thandle, Thandle);
Terr     EXPORT_FN     TMG_JPEG_sequence_calc_length(Thandle);
Terr     EXPORT_FN     TMG_JPEG_sequence_extract_frame(Thandle, Thandle, ui32);
Terr     EXPORT_FN     TMG_JPEG_sequence_set_start_frame(Thandle, ui32);
Terr     EXPORT_FN     TMG_JPEG_set_file_ptr(Thandle, FILE*);  /* non-doc/user */
Terr     EXPORT_FN     TMG_JPEG_set_image(Thandle, Thandle);

/*
 * TMG_jpeg.c
 * ----------
TMG_jpeg.c(ForBrief)
 */
Terr EXPORT_FN TMG_JPEG_compress(Thandle, Thandle, ui16);
Terr EXPORT_FN TMG_JPEG_decompress(Timage_handle, Thandle, ui16);
Terr EXPORT_FN TMG_JPEG_set_Quality_factor(Thandle, ui16);
Terr EXPORT_FN TMG_JPEG_set_Quantization_factor(Thandle, ui16);
Terr EXPORT_FN TMG_JPEG_ConvertToMonoYUV( Thandle hJpegImage);   /* By setting chrominance Q table to 255s */
Terr EXPORT_FN TMG_JPEG_make_Q_tables(Thandle);                  /* non-doc/user */
Terr EXPORT_FN TMG_JPEG_set_default_H_tables(Thandle);           /* non-doc/user */
Terr EXPORT_FN TMG_JPEG_generate_H_codetable_LUTs(Thandle);      /* non-doc/user */
Terr EXPORT_FN TMG_JPEG_generate_Huffman_decode_tables();        /* non-doc/user */

/*
 * TMG_cmap.c - Optimum palette generation.
 * ----------
TMG_cmap.c(ForBrief)
 */
#if defined _TMG_MAC
EXPORT_FN struct Tcmap_info* TMG_cmap_create_info(void);             /* non-doc/user */
#else
struct Tcmap_info EXPORT_FN_PTR TMG_cmap_create_info(void);          /* non-doc/user */
#endif
Terr     EXPORT_FN  TMG_cmap_destroy_info(Thandle);                  /* non-doc/user */
Terr     EXPORT_FN  TMG_convert_to_paletted(Thandle, Thandle, ui16); /* non-doc/user */
Terr     EXPORT_FN  TMG_cmap_generate(Thandle, ui16, ui16);
ui32     EXPORT_FN  TMG_cmap_get_occurrences(Thandle, ui16);
ui32     EXPORT_FN  TMG_cmap_get_RGB_colour(Thandle, ui16);
ui8      EXPORT_FN  TMG_cmap_find_closest_colour(Thandle, ui8, ui8, ui8);
Terr     EXPORT_FN  TMG_cmap_generate_cmap_info(Thandle);
Tboolean EXPORT_FN  TMG_cmap_is_grayscale(Thandle);
Terr     EXPORT_FN  TMG_cmap_set_colour(Thandle, ui16, ui16);
Terr     EXPORT_FN  TMG_cmap_set_RGB_colour(Thandle, ui16, ui8, ui8, ui8);
Terr     EXPORT_FN  TMG_cmap_set_type(Thandle, ui16);
Tboolean EXPORT_FN  TMG_image_is_colour(Thandle);

/*
 * TMG_scl.c - Image scaling and sub-sampling routines.
 * ----------
TMG_scl.c(ForBrief)
 */
Terr EXPORT_FN TMG_IP_subsample(ui32 hInImage, ui32 hOutImage, int nFactor, ui16 TMG_action);
Terr EXPORT_FN TMG_IP_pixel_rep(Thandle, Thandle, ui16, ui16);
Terr EXPORT_FN TMG_IP_bilinear_zoom( Thandle Hin_image, Thandle Hout_image, i32 *piRoi, ui16 XSize, ui16 YSize, ui16 *pwXCoeffs, ui16 *pwYCoeffs, ui16 TMG_action );
Terr EXPORT_FN TMG_IP_bilinear_zoom_setup(i32*, ui16, ui16, ui16*, ui16*);

/*
 * TMG_raw.c - Raw data write & read.
 * ---------
TMG_raw.c(ForBrief)
 */
Terr EXPORT_FN TMG_write_raw_data_file(Thandle, ui16);

/*
 * TMG_rot.c - TMG image rotation.
 * ---------
TMG_rot.c(ForBrief)
 */
Terr EXPORT_FN TMG_image_rotate(Thandle, Thandle, ui16, ui16); /* Note: not in current libraries */

/*
 * TMG_rw.c - TMG image read/write wrapper.
 * --------
TMG_rw.c(ForBrief)
 */
Terr EXPORT_FN TMG_image_find_file_format(char*);
Terr EXPORT_FN TMG_image_read(Thandle, Thandle, ui16);
Terr EXPORT_FN TMG_image_write(Thandle, Thandle, ui16, ui16);

/*
 * TMG_bay.c - TMG Bayer functions.
 * ---------
TMG_bay.c(ForBrief)
 */
Terr EXPORT_FN TMG_BAY_RGB24_to_RGGB32(Thandle Hin_image, Thandle Hout_image, ui16 TMG_action);
Terr EXPORT_FN TMG_BAY_RGGB32_map_to_RGB24(Thandle Hin_image, Thandle Hout_image, int nPixelOrigin, ui16 TMG_action);
Terr EXPORT_FN TMG_BAY_RGGB32_map_to_Y8(Thandle Hin_image, Thandle Hout_image, ui16 TMG_action);
Terr EXPORT_FN TMG_BAY_RGGB32_to_BGRX32(Thandle Hin_image, Thandle Hout_image, int nDecodeScheme, int nTopLeftPixel, ui16 TMG_action);
Terr EXPORT_FN TMG_BAY_RGGB32_to_RGB16(Thandle Hin_image, Thandle Hout_image, ui16 wDecodeScheme, ui16 wPixelCol, ui16 TMG_action);
Terr EXPORT_FN TMG_BAY_ColorBalance(Thandle hInImage, Thandle hOutImage, ui8 bR, ui8 bG, ui8 bB, ui16 TMG_action);

/*
 * TMG_draw.c - TMG Draw / Timestamp functions.
 * ---------
TMG_draw.c(ForBrief)
 */
Terr EXPORT_FN TMG_DrawText(Thandle hImage, char *pszString, struct sTMG_Font *psTmgFont, i32 nX, i32 nY, ui32 dwMode, ui32 dwColour);

Terr EXPORT_FN TMG_draw_get_ptr(ui32 dwType, void **ppData);
Terr EXPORT_FN TMG_draw_text(Thandle hImage, char *pszString, struct sTMG_Font *psTmgFont, ui32 dwX, ui32 dwY, ui32 dwMode, ui32 dwColour);
Terr EXPORT_FN TMG_draw_timestamp(Thandle hImage, struct sTMG_Font *psTmgFont, ui32 dwX, ui32 dwY, ui32 dwMode, ui32 dwColour);
ui32 EXPORT_FN TMG_DrawPixel(ui32 hImage, ui32 dwColour, i32 nX, i32 nY);
Terr EXPORT_FN TMG_DrawCircle( ui32 hImage, i32 nX, i32 nY, i32 nRadius, ui32 dwColor, ui32 dwThickness, ui32 dwMode);
Terr EXPORT_FN TMG_DrawLine( ui32 hImage, i32 nX1, i32 nY1, i32 nX2, i32 nY2, ui32 dwColor, ui32 dwThickness, ui32 dwMode);
Terr EXPORT_FN TMG_DrawBox( ui32 hImage, i32 nX1, i32 nY1, i32 nX2, i32 nY2, ui32 dwColor, ui32 dwThickness, ui32 dwMode);
Terr EXPORT_FN TMG_DrawBoxFill( ui32 hImage, i32 nX1, i32 nY1, i32 nX2, i32 nY2, ui32 dwColor, ui32 dwMode);


/*
 * TMG_spl.c - TMG special functions and image processing functions.
 * ---------
TMG_spl.c(ForBrief)
 */
Terr EXPORT_FN TMG_IP_ShadingCorrection_Generate(ui32 hBlackRefImage, ui32 hWhiteRefImage, ui32 hOffsetImage, ui32 hGainImage);
Terr EXPORT_FN TMG_IP_ShadingCorrection(ui32 hInImage, ui32 hOutImage, ui32 hOffsetImage, ui32 hGainImage);
Terr EXPORT_FN TMG_IP_extract_region(Thandle Hin_image, Thandle Hout_image, ui32 dwRegionType, ui32 dwX, ui32 dwY, ui32 dwRadius, ui16 TMG_action);
Terr EXPORT_FN TMG_IP_filter_3x3(Thandle, Thandle, i32*, ui16);
Terr EXPORT_FN TMG_IP_generate_averages(Thandle Hin_image, struct tTMG_Averages *psAverages, ui16 TMG_action);
Terr EXPORT_FN TMG_IP_histogram_clear(struct tTMG_Histogram *psHistogram);
Terr EXPORT_FN TMG_IP_histogram_copy(struct tTMG_Histogram *psHistogramIn, struct tTMG_Histogram *psHistogramOut);
Terr EXPORT_FN TMG_IP_histogram_filter(struct tTMG_Histogram *psInHistogram, i32 nFilterOrder);
Terr EXPORT_FN TMG_IP_histogram_generate(Thandle Hin_image, struct tTMG_Histogram *psHistogram, ui16 TMG_action);
Terr EXPORT_FN TMG_IP_histogram_match(struct tTMG_Histogram *psRefHistogram, struct tTMG_Histogram *psInHistogram, i32 n32Plane, ui32 *pdwResult);
Terr EXPORT_FN TMG_IP_mirror(Thandle hInImage, Thandle hOutImage, i32 *roi, ui16 TMG_action);
Terr EXPORT_FN TMG_IP_mirror_image(Thandle, Thandle, ui16);
Terr EXPORT_FN TMG_IP_motion_detect(Thandle hCurrentImage, Thandle hLastImage, i32 nThreshold, i32 nStride, i32 *pnPercentChange);
Terr EXPORT_FN TMG_IP_rotate_image(Thandle, Thandle, ui32);
Terr EXPORT_FN TMG_IP_scale(Thandle Hin_image, Thandle Hout_image, ui32 dwNewWidth, ui32 dwNewHeight, ui16 TMG_action);
Terr EXPORT_FN TMG_IP_threshold_grayscale(Thandle, Thandle, ui8, ui8, ui16);
Terr EXPORT_FN TMG_SPL_2fields_to_frame(Thandle, Thandle, Thandle, ui16);
Terr EXPORT_FN TMG_SPL_Data32_to_Y8(Thandle Hin_image, Thandle Hout_image, ui16 wShiftRight, ui16 TMG_action);
Terr EXPORT_FN TMG_SPL_field_to_frame(ui32 hInImage, ui32 hOutImage, ui16 wInterpScheme, ui16 TMG_action);
Terr EXPORT_FN TMG_SPL_HSI_to_RGB_pseudo_colour(Thandle Hin_image, Thandle Hout_image, ui16 TMG_action);
Terr EXPORT_FN TMG_SPL_interlace_frame(Thandle, Thandle, ui16);
Terr EXPORT_FN TMG_SPL_XXXX32_to_Y8(Thandle, Thandle, ui16, ui16);
Terr EXPORT_FN TMG_SPL_Y8_to_pseudo_colour(Thandle, Thandle, ui16, ui16);
Terr EXPORT_FN TMG_SPL_YUV422_to_RGB_pseudo_colour(Thandle Hin_image, Thandle Hout_image, ui16 TMG_action);

/*
 * TMG_test.c - Various development test programs.
 * ----------
TMG_test.c(ForBrief)
 */
Terr  EXPORT_FN  TMG_print_Huffman_tables(Thandle);  /* non-doc/user */
Terr  EXPORT_FN  TMG_print_image(Thandle, char*);    /* non-doc/user */
Terr  EXPORT_FN  TMG_print_cmap(Thandle, char*);     /* non-doc/user */
Terr  EXPORT_FN  TMG_print_jpeg_image(Thandle);      /* non-doc/user */
Terr  EXPORT_FN  TMG_test_pattern(Thandle, ui16);    /* non-doc/user */
Terr  EXPORT_FN  TMG_JPEG_Q_table_to_zero(Thandle Hjpeg_image);  /* non-doc/user */
Terr  EXPORT_FN  TMG_JPEG_Q_table_print(Thandle Hjpeg_image);    /* non-doc/user */

/*
 * TMG_tga.c - TGA reader and writer
 * ---------
TMG_tga.c(ForBrief)
 */
Terr EXPORT_FN TMG_write_TGA_file(Thandle, ui16);  /* non-doc */
Terr EXPORT_FN TMG_read_TGA_file(Thandle, ui16);   /* non-doc */

/*
 * TMG_tiff.c - TIFF reader and writer
 * ----------
TMG_tiff.c(ForBrief)
 */
Terr EXPORT_FN TMG_write_TIFF_file(Thandle, ui16); /* non-doc */
Terr EXPORT_FN TMG_read_TIFF_file(Thandle, ui16);  /* non-doc */
ui16 EXPORT_FN TIFF_read_ui16(FILE*, ui16);        /* non-doc/user */
ui32 EXPORT_FN TIFF_read_ui32(FILE*, ui16);        /* non-doc/user */
ui32 EXPORT_FN TIFF_read_data(FILE*, ui16, ui16);  /* non-doc/user */

/*
 * TMG_chrm.c - Chroma keying and related functions.
 * ----------
TMG_chrm.c(ForBrief)
 */
Thandle EXPORT_FN TMG_CK_create(void);
Terr    EXPORT_FN TMG_CK_destroy(Thandle);
Terr    EXPORT_FN TMG_CK_chroma_key(Thandle, Thandle, Thandle, ui16, Thandle, ui16, ui16);
Terr    EXPORT_FN TMG_CK_calibrate(Thandle, Thandle, ui16);
Terr    EXPORT_FN TMG_CK_set_parameter(Thandle, ui16, ui16);
Terr    EXPORT_FN TMG_CK_get_parameter(Thandle, ui16);
ui8     EXPORT_FN TMG_CK_get_component(ui16, ui16);
ui32    EXPORT_FN TMG_CK_get_YUV_values(ui16);
ui32    EXPORT_FN TMG_CK_get_YUV_values_RGB(ui16, ui16, ui16);
Terr    EXPORT_FN TMG_CK_generate_UV_to_hue_LUT(void);
Terr    EXPORT_FN TMG_CK_generate_UV_to_hue_LUT_TC06(void);
Terr    EXPORT_FN TMG_CK_destroy_UV_to_hue_LUT(void);


/*
 * TMG_mjpeg.c - Motion JPEG functions.
 * -----------
TMG_mjpeg.c(ForBrief)
 */
ui32 EXPORT_FN     TMG_MJPEG_Create( ui32 *pdwHandle, char *pszFilename, i32 nCameraNumber, i32 nQFactor);
ui32 EXPORT_FN     TMG_MJPEG_Open( ui32 *phHandle, char *pszFilename);
ui32 EXPORT_FN     TMG_MJPEG_GetInfo( ui32 hHandle, struct tTMG_MJPEG *psTmgMjpeg);
ui32 EXPORT_FN     TMG_MJPEG_SetInfo( ui32 hHandle, struct tTMG_MJPEG *psTmgMjpeg);
ui32 EXPORT_FN     TMG_MJPEG_Close( ui32 hHandle);
ui32 EXPORT_FN     TMG_MJPEG_CopyParams( ui32 hMjpegHandleIn, ui32 hMjpegHandleOut);
ui32 EXPORT_FN     TMG_MJPEG_ImageRead( ui32 hHandle, ui32 hImage, i32 nFrameNum);
ui32 EXPORT_FN     TMG_MJPEG_ScanForwards( ui32 hHandle, i32 nNumFrames);
ui32 EXPORT_FN     TMG_MJPEG_ScanBackwards( ui32 hHandle, i32 nNumFrames);
ui32 EXPORT_FN     TMG_MJPEG_SkipForwards( ui32 hHandle, i32 nNumFrames);
ui32 EXPORT_FN     TMG_MJPEG_SkipBackwards( ui32 hHandle, i32 nNumFrames);
ui8  EXPORT_FN_PTR TMG_MJPEG_GetNextMemoryBuffer( ui32 hHandle);
ui32 EXPORT_FN     TMG_MJPEG_IndexesAdd( ui32 hHandle);
ui32 EXPORT_FN     TMG_MJPEG_ImageWrite( ui32 hHandle, ui32 hImage);

ui32 EXPORT_FN TMG_MJPEG_CommentStringGenerate( struct tTMG_MJPEG *psMJPEG);
ui32 EXPORT_FN TMG_MJPEG_CommentStringParse( struct tTMG_MJPEG *psMJPEG);
ui32 EXPORT_FN TMG_MJPEG_InitLibrary(void);
ui32 EXPORT_FN TMG_MJPEG_BufferWrite(Timage_handle Hjpeg_image, ui8 *pbData, ui32 *pdwCount, ui8 *pszCommentString);
ui32 EXPORT_FN TMG_MJPEG_DEBUG_DumpData( struct tTMG_MJPEG *psMJPEG, char *pszFilename);

/*
 * TMG_mraw.c - Motion RAW functions (sequence recording).
 * ----------
TMG_mraw.c(ForBrief)
 */
ui32 EXPORT_FN  TMG_MRAW_InitLibrary(void);
ui32 EXPORT_FN  TMG_MRAW_Create( ui32 *phHandle, char *pszFilename);
ui32 EXPORT_FN  TMG_MRAW_GetInfo( ui32 hHandle, struct tTMG_MRAW *psTmgMraw);
ui32 EXPORT_FN  TMG_MRAW_SetInfo( ui32 hHandle, struct tTMG_MRAW *psTmgMraw);
ui32 EXPORT_FN  TMG_MRAW_Close( ui32 hHandle);
ui32 EXPORT_FN  TMG_MRAW_ImageWrite( ui32 hHandle, ui32 hImage);
ui32 EXPORT_FN  TMG_MRAW_HeaderPerFile_Generate( struct tTMG_MRAW *psMRAW);
ui32 EXPORT_FN  TMG_MRAW_HeaderPerImage_Generate( struct tTMG_MRAW *psMRAW);

ui32 EXPORT_FN  TMG_MRAW_Open( ui32 *phHandle, char *pszFilename);
ui32 EXPORT_FN  TMG_MRAW_OpenFast( ui32 *phHandle, char *pszFilename);
ui32 EXPORT_FN  TMG_MRAW_HeaderPerFile_Read( struct tTMG_MRAW *psMRAW);
ui32 EXPORT_FN  TMG_MRAW_HeaderPerImage_Read( struct tTMG_MRAW *psMRAW);
ui32 EXPORT_FN  TMG_MRAW_ImageRead( ui32 hHandle, ui32 hImage, i32 nFrameNum);
ui32 EXPORT_FN  TMG_MRAW_SkipFrames( ui32 hHandle, i32 nNumFrames);

/*
 * TMG_lut.c - LUT image editing functions - e.g. contrast, gamma etc.
 * ---------
TMG_lut.c(ForBrief)
 */
Thandle EXPORT_FN     TMG_LUT_create(ui32, ui16);
Terr    EXPORT_FN     TMG_LUT_destroy(Thandle);
Terr    EXPORT_FN     TMG_LUT_apply(Thandle, Thandle, Thandle, ui16);
Terr    EXPORT_FN     TMG_LUT_generate(Thandle, i16, i16, i16, i16, i16, i16);
void    EXPORT_FN_PTR TMG_LUT_get_ptr(Thandle, ui16);


/*
 * TMG_os.c - OS dependent functions.
 * --------
 */
ui32 EXPORT_FN TMG_CriticalSectionInit(void);
ui32 EXPORT_FN TMG_CriticalSectionBegin(void);
ui32 EXPORT_FN TMG_CriticalSectionEnd(void);
#if defined _TMG_WINDOWS
INT TMG_CriticalSectionhandler(LPEXCEPTION_POINTERS pExceptionRecord, const char *szFnName );
#elif defined _TMG_LINUX

#endif



#ifdef __cplusplus
};
#endif


/*
 * Externals Definitions
 * ---------------------
 *
 * Global Array of Snapper Handles
Externals(ForBrief)
 */
extern struct tTMG gsTMG_Internal[];

extern volatile int gfInitialised;

extern struct Timage *Pimage_array[];
extern struct Ttmg_LUT *pLUT_array[];
extern struct Tdisplay *Pdisplay_array[];
extern struct Tchroma_key *Pchroma_key_array[];

/*
 * YUV to 16 bit RGB LUT for fast colour display.
 * And RGB16/Y8 to paletted for fast paletted display.
 */
extern ui8  *RGB16_to_paletted_LUT;
extern ui8  *Y8_to_paletted_LUT;
extern ui8  *Y16_to_paletted_LUT;
extern ui8  *YUV_to_paletted_LUT;
extern ui16 *YUV_to_RGB16_LUT;
extern ui16 *YUV_to_RGB15_LUT;


/*
 * UV to hue (angle) LUT.
 */
extern ui16 *UV_to_hue_LUT;

/*
 * Software JPEG Huffman decode LUTs (used in tmg_jpeg.c).
 */
extern struct Tdecodetable *gsAC_Y_decode;
extern struct Tdecodetable *gsAC_C_decode;


/* Other JPEG related stuff
 * ------------------------
 * See also #define TMG_JPEG_JFIF_LENGTH.
 */
extern char TMG_JPEG_JFIF_String[];


/* Malloc Debugging
 * ----------------
 */
extern i32 gnTmgDebugMallocCount;
extern i32 gnTmgDebugFreeCount;
extern i32 gnTmgDebugTotalMalloced;


/* MJPEG Debugging
 * ---------------
 */
extern i32 gfTmgMjpegDebug;


/*
 * OS dependent globals
 * --------------------
 */
#if defined _TMG_WINDOWS
extern CRITICAL_SECTION gsTMG_CriticalSection;

#elif defined _TMG_LINUX
extern pthread_mutexattr_t gMutexAttr;
extern pthread_mutex_t gDeviceMutex;

#elif defined _TMG_MACOSX
extern pthread_mutexattr_t gMutexAttr;
extern pthread_mutex_t gDeviceMutex;

#endif  /* OS dependent code */

#endif  /* _TMG_PRO_H_ */
