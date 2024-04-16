from atmcd import atmcd

print("Spool Example")

print("Intialising Camera")
sdkObject = atmcd() #load the atmcd library

(ret) = sdkObject.Initialize("") #initialise camera
print("Initialize returned",ret)

if atmcd.DRV_SUCCESS==ret:
    
  (ret, iSerialNumber) = sdkObject.GetCameraSerialNumber()
  print("GetCameraSerialNumber returned:",ret,"Serial No:",iSerialNumber)
  
  #configure the acquisition
  (ret) = sdkObject.CoolerON()
  print("Function CoolerON returned",ret)
    
  (ret) = sdkObject.SetAcquisitionMode(3)
  print("Function SetAcquisitionMode returned",ret,"mode = Kinetic Series")
  
  (ret) = sdkObject.SetReadMode(4)
  print("Function SetReadMode returned",ret,"mode = Image")
  
  (ret) = sdkObject.SetTriggerMode(0)
  print("Function SetTriggerMode returned",ret,"mode = Internal")
  
  (ret, xpixels, ypixels) = sdkObject.GetDetector()
  print("Function GetDetector returned",ret,"xpixels =",xpixels,"ypixels =",ypixels)

  (ret) = sdkObject.SetImage(1, 1, 1, xpixels, 1, ypixels)
  print("Function SetImage returned",ret,"hbin = 1 vbin = 1 hstart = 1 hend =",xpixels,"vstart = 1 vend =",ypixels)
  
  (ret) = sdkObject.SetNumberKinetics(5)
  print("Function SetNumberKinetics returned",ret)
  
  (ret) = sdkObject.SetExposureTime(0.001)
  print("Function SetExposureTime returned",ret,"time = 0.01s")
  
  (ret, fminExposure, fAccumulate, fKinetic) = sdkObject.GetAcquisitionTimings()
  print("Function GetAcquisitionTimings returned",ret,"exposure =",fminExposure,"accumulate =",fAccumulate,"kinetic =",fKinetic)

  file_spooling = "C:\VM\Spool"
  (ret) = sdkObject.SetSpool(1, 7, file_spooling, 0)
  print("Function SetSpool returned",ret)  

  (ret) = sdkObject.PrepareAcquisition()
  print("Function PrepareAcquisition returned",ret)
  
  #Perform Acquisition
  (ret) = sdkObject.StartAcquisition()
  print("Function StartAcquisition returned",ret)
  
  last = 0
  while (last < 5):
      (ret) = sdkObject.WaitForAcquisition()
      print("Function WaitForAcquisition returned",ret)
      
      (ret, first, last) = sdkObject.GetNumberNewImages()
      print("Function GetNumberNewImages returned",ret,"first =",first,"last =",last)
                          
      imageSize = xpixels * ypixels
      (ret, fullFrameBuffer, validfirst, validlast) = sdkObject.GetImages(first, last, imageSize)
      print("Function GetImages returned",ret,"first pixel =",fullFrameBuffer[0],"size =",imageSize) 
      print(fullFrameBuffer)  

  #Clean up
  (ret) = sdkObject.CoolerOFF()
  (ret) = sdkObject.GetTemperature()
  print("Temperature ", ret)
  (ret) = sdkObject.ShutDown()
  print("Shutdown returned",ret)

else:
  print("Cannot continue, could not initialise camera")
  
