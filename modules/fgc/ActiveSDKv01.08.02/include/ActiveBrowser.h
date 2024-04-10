#pragma once

#if defined ActiveBrowser_EXPORTS
#define ACTIVEBROWSER_DECL_EXPORT __declspec(dllexport) __cdecl
#else
#define ACTIVEBROWSER_DECL_EXPORT __declspec(dllimport) __cdecl
#endif // ACTIVEBROWSER_EXPORT

/*!
* \brief The ActiveBrowser library allows using the ActiveCapture feature
browser in your own applications. The widget can display any number of GenApi
nodemaps, each in their own tab. ActiveCapture being a Qt application the
library provides a way to integrate the widget within an MFC program.

   The ActiveBrowser Qt class is currently not exported, but will be in a later
revision of the library.

   MFC users
   *********
   ActiveCapture is developed using the Qt framework, which is not directly
compatible with MFC. Therefore the library provides an API to use ActiveBrowser
from an MFC application. The following Qt dependencies must be located in the
application's directory. Those are installed with ActiveCapture, via the
FireBird installer and you will need to ensure they can be found by your
application; they are located in the installation directory next to the
ActiveBrowser library:
   - Qt5Core.dll
   - Qt5Gui.dll
   - Qt5Widgets.dll
   - Qt5PrintSupport.dll
   - libGLESV2.dll
   - platforms
     - qwindows.dll
   - iconengines

   MFC dialog-based and MDI/SDI programs
   *************************************
   The library should be initialised differently depending on whether the MFC
application is dialog-based or not. Dialog-based MFC programs should use
ActiveBrowser_CreateQApp() and ActiveBrowser_DestroyQApp() whereas MDI/SDI MFC
programs should use ActiveBrowser_MfcRun().

   Note about MFC's memory leak detection
   **************************************
   MFC contains a checkpoint-based memory leak detection mechanism.
   This mechanism does not handle well Qt's system of allocating global static
objects. The result is that when running applications that combine Qt and MFC
(like the examples below) from within Visual Studio, one will get a report about
leaked objects upon application exit. These warnings can safely be ignored.

   Note about memory footprint
   ***************************
   The usage of icons in the ActiveBrowser, internally requires the creation of
a QIconEngine. This, in turn, creates a static instance of a
QMimeDatabasePrivate. This instance being static, it persists even after
ActiveBrowser_Destroy() has been called. This is normal and should not be
considered a memory leak.
*/

#include <GenICam.h>
#include <windows.h>

// Forward declaration
class CWinApp;

extern "C" {
enum ActiveBrowser_GenTLModuleType {
    GenTLModuleType_System,
    GenTLModuleType_Interface,
    GenTLModuleType_Device,
    GenTLModuleType_Stream,
    GenTLModuleType_RemoteDevice
};

enum ActiveBrowser_EventType {
    EventType_Closed =
        0 //! The ActiveBrowser close button has been clicked. The ActiveBrowser
          //! does not get destroyed on close and this is a good time to call
          //! ActiveBrowser_Destroy().
};

struct IActiveBrowser {
    /*!
    * \brief Add a feature browser treeview widget to the main ActiveBrowser.
    * \param ModuleType GenTL module type (System, Interface, etc.).
    This is used for two things:
    - Select the appropriate icon and name for the tab in which the treeview is
    shown.
    - Optionally synchronize certain features between the Remote Device and the
    Device nodemaps.
    * \param Name A name to identify the nodemap that will be used in error
    messages.
    * \param Nodemap The nodemap that should be controlled by the treeview.
    */
    virtual void AddNodemap(ActiveBrowser_GenTLModuleType ModuleType,
                            const char *Name, GenApi::CNodeMapRef Nodemap) = 0;

    /*!
    * \brief Enables or disables glue logic synchronizing features between
    Remote Device and Device nodemaps.
    * ActiveBrowser synchronizes certain features of the Remote Device's
    nodemap, with the equivalent features of the Device's nodemap. This improves
    usability as it removes the need for the user to mirror changes made to one
    nodemap onto the other. The following features of the Device nodemap are
    synchronized with the relevant features of the Device's nodemap: Width,
    Height, PixelFormat, DeviceTapGeometry, DeviceScanType and AcquisitionMode.
    Note: calling this function will trigger read and write accesses to the
    camera and the frame grabber.
    */
    virtual void EnableGlue() = 0;

    /*!
    * \brief Causes the ActiveBrowser dialog to become visible. Clicking the
    cross button on the top-right corner of the dial hides it.
    */
    virtual void Show() = 0;

    /*!
     * \brief Configure the ActiveBrowser dialog to either be a modal or
     * modeless dialog. Must be called prior to Show().
     */
    virtual void SetModal(bool Modal) = 0;

    /*!
     * \brief Callback function called by IActiveBrowser to signal asynchronous
     * events. \param pPrivate Pointer to user data as passed in the
     * SetMessageHandler(). \param MsgType Represents the event type. \param
     * Data Optional data for the event.
     */
    typedef void (*MessageHandler)(void *pPrivate,
                                   enum ActiveBrowser_EventType MsgType,
                                   void *Data);

    /*!
     * \brief Set a callback used by IActiveBrowser to signal asynchronous
     * events. \param pFct The callback function. \param pPrivate Pointer to
     * user data that is passed as argument when the callback is called.
     */
    virtual void SetMessageHandler(MessageHandler pFct, void *pPrivate) = 0;

    virtual ~IActiveBrowser() {}
};

/*!
 * \brief Runs the event loop for both Qt and the MFC application object
 mfcApp, and returns the result. This function creates a QApplication
 object and starts the event loop. It deletes the created QApplication
 before returning.

 This function should be called in a reimplementation of CWinApp::Run():

 \code
 int MyMfcApp::Run()
 {
 return QMfcApp::run(this);
 }
 \endcode

 Since a QApplication object must exist before Qt widgets can be
 created you cannot use this function if you want to use Qt-based
 user interface elements in, the InitInstance() function of CWinApp (for example
 for dialog-based MFC program that call the CDialog::DoModal() in
 CWinApp::InitInstance()). Such programs should instead use
 ActiveBrowser_CreateQApp().

 * \param mfcApp Pointer to the CWinApp instance.
 * \return An int value that is returned by WinMain.
*/
int ACTIVEBROWSER_DECL_EXPORT ActiveBrowser_MfcRun(CWinApp *mfcApp);

/*!
* \brief Create the QApplication instance required by ActiveBrowser. This
function should be used when ActiveBrowser_MfcRun() cannot be used. This is the
case when ActiveBrowser is used from within a dialog-based MFC program and the
CDialog::DoModal() function is used in the CWinApp::InitInstance() function
(CWinApp::Run() is not called). ActiveBrowser_DestroyQApp() must be called to
destroy the created QApplication instance.
* \param mfcApp Pointer to the CWinApp instance.
*/
void ACTIVEBROWSER_DECL_EXPORT ActiveBrowser_CreateQApp(CWinApp *mfcApp);

/*!
 * \brief Delete the QApplication instance created via
 * ActiveBrowser_CreateQApp().
 */
void ACTIVEBROWSER_DECL_EXPORT ActiveBrowser_DestroyQApp();

/*!
 * \brief Typedef for the error handler function. Msg is invalid once the
 function has returned, so must be copied before.
*/
typedef void (*ErrorHandlerType)(const char *Msg);

void ACTIVEBROWSER_DECL_EXPORT
ActiveBrowser_DefaultErrorHandler(const char *Msg);

/*!
 * \brief Set the error handler that ActiveBrowser will use to report errors.
   Should be called once prior to any other function.
 * \param fct The function pointer.
*/
void ACTIVEBROWSER_DECL_EXPORT
ActiveBrowser_SetErrorHandler(ErrorHandlerType fct);

/*!
 * \brief Creates an instance of the ActiveBrowser object.
 * \param parent Handle to the parent Win32 window.
 * \param out The created ActiveBrowser object.
 * \return void ACTIVEBROWSER_DECL_EXPORT
 */
void ACTIVEBROWSER_DECL_EXPORT
ActiveBrowser_Create(HWND parent, struct IActiveBrowser **out);

/*!
 * \brief Deletes an ActiveBrowser instance previously created with
 * ActiveBrowser_Create(). \param in The ActiveBrowser instance to destroy.
 * \return void ACTIVEBROWSER_DECL_EXPORT
 */
void ACTIVEBROWSER_DECL_EXPORT ActiveBrowser_Destroy(IActiveBrowser *in);
}