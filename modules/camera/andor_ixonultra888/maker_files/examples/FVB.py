from pyAndorSDK2.atmcd import atmcd

sdk = atmcd() #load the atmcd library

(ret) = sdk.Initialize("") #initialise camera
print("Initialize returned",ret)

if atmcd.DRV_SUCCESS==ret:
    
  (ret, iSerialNumber) = sdk.GetCameraSerialNumber()
  print("GetCameraSerialNumber returned:",ret,"Serial No:",iSerialNumber)
  
  #configure the acquisition
  (ret) = sdk.CoolerON()
  print("Function CoolerON returned",ret)
    
  (ret) = sdk.SetAcquisitionMode(1)
  print("Function SetAcquisitionMode returned",ret,"mode = Single Scan")
  
  (ret) = sdk.SetReadMode(0)
  print("Function SetReadMode returned",ret,"mode = FVB")
  
  (ret) = sdk.SetTriggerMode(0)
  print("Function SetTriggerMode returned",ret,"mode = Internal")
  
  (ret, xpixels, ypixels) = sdk.GetDetector()
  print("Function GetDetector returned",ret,"xpixels =",xpixels,"ypixels =",ypixels)

  (ret) = sdk.SetImage(1, 1, 1, xpixels, 1, ypixels)
  print("Function SetImage returned",ret,"hbin = 1 vbin = 1 hstart = 1 hend =",xpixels,"vstart = 1 vend =",ypixels)

  (ret) = sdk.SetExposureTime(0.01)
  print("Function SetExposureTime returned",ret,"time = 0.01s")
  
  (ret, fminExposure, fAccumulate, fKinetic) = sdk.GetAcquisitionTimings()
  print("Function GetAcquisitionTimings returned",ret,"exposure =",fminExposure,"accumulate =",fAccumulate,"kinetic =",fKinetic)

  (ret) = sdk.PrepareAcquisition()
  print("Function PrepareAcquisition returned",ret)
  
  #Perform Acquisition
  (ret) = sdk.StartAcquisition()
  print("Function StartAcquisition returned",ret)

  (ret) = sdk.WaitForAcquisition()
  print("Function WaitForAcquisition returned",ret)
                      
  imageSize = xpixels
  (ret, fullFrameBuffer) = sdk.GetMostRecentImage(imageSize)
  print("Function GetMostRecentImage returned",ret,"first pixel =",fullFrameBuffer[0],"size =",imageSize)   

  #Clean up
  (ret) = sdk.ShutDown()
  print("Shutdown returned",ret)

else:
  print("Cannot continue, could not initialise camera")
  
