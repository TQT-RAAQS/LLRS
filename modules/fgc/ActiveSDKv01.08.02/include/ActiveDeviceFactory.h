#pragma once

#if defined ActiveDeviceFactory_EXPORTS
#define ACTIVEDEVICEFACTORY_DECL_EXPORT __declspec(dllexport) __cdecl
#else
#define ACTIVEDEVICEFACTORY_DECL_EXPORT __declspec(dllimport) __cdecl
#endif //ActiveDeviceFactory_EXPORTS

/*!
* \brief The ActiveDeviceFactory provides an API to enumerate the cameras in the system,
* to open one and retrieve a GenICam nodemap to control it.
* The library acts as a GenTL consumer and therefore requires a GenTL Producer to 
* be installed as specified in the GenTL standard or to be found in the local working
* directory.
* 
* Qt Dependency
* *************
* The following Qt dependencies must be located in the application's directory. Those are 
* installed with ActiveCapture, via the FireBird installer; they are located in the FireBird 
* installation directory:
*   - Qt5Core.dll
*   - platforms
*      - qwindows.dll
*/

#include <GenICam.h>
#include <GenTL.h>
#include <TLActiveSilicon.h>

#include <phx_api.h>

extern "C" {

   struct IActiveDevice
   {
      /*!
       * \brief Retrieve the nodemap for the camera.
       * \return The nodemap giving access to the features of the camera or an empty nodemap 
       * if the function fails. In case of failure, the error handler is called. 
       * CoaXPress: The function should always succeeds so long as the camera 
       * has auto-discovered. 
       * Camera Link: Make sure to run the "GenTL CL Setup Utility" so that the 
       * GenTL Producer can discover the camera. The COM port to the camera must not already
       * be in use (PHX handle should be open as PHX_ACQ_ONLY).
      */
      virtual GenApi::CNodeMapRef GetRDevNodemap()= 0;
   };

   struct IActiveDeviceFactory
   {
      /*!
       * \brief Refresh the list of GenTL Modules (Systems, Interfaces, Devices and Streams)
       * known to the Factory.
      */
      virtual void RefreshKnownDeviceList() = 0;

      /*!
       * \brief Open a device by PHX handle. Internally calls LoadLibrary() passing-in the 
       * module name for the PHX library without a path, thus using the module already
       * loaded for the PHX library. 
       * \param hPhx A valid PHX library handle to the board / channel to open.
       * \return IActiveDevice * Pointer to the device or NULL if the device could
       * not be opened.
      */
      virtual IActiveDevice *OpenDevice(tHandle hPhx) = 0;

      /*!
       * \brief Close the device and deletes the IActiveDevice instance.
       * \param pDevice Pointer to the Device to close.
      */
      virtual void CloseDevice(IActiveDevice *pDevice) = 0;
   };

   /*!
   * \brief Typedef for the error handler function.
   */
   typedef void(*ErrorHandlerType)(const char *szMsg);

   /*!
    * \brief Set the error handler that ActiveDeviceFactory will use to report errors. 
    * 
    * Should be called once prior to any other function.
    * \param fct The function pointer.
   */
   void ACTIVEDEVICEFACTORY_DECL_EXPORT ActiveDeviceFactory_SetErrorHandler(ErrorHandlerType fct);

   /*!
    * \brief Returns the device factory.
   */
   void ACTIVEDEVICEFACTORY_DECL_EXPORT ActiveDeviceFactory_Get(IActiveDeviceFactory **);
}