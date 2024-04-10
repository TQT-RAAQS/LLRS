from pyAndorSDK2.atmcd import atmcd

sdk = atmcd() #load the atmcd library

(ret) = sdk.Initialize("") #initialise camera
print("Initialize returned",ret)

if atmcd.DRV_SUCCESS==ret:
    
  (ret, iSerialNumber) = sdk.GetCameraSerialNumber()
  print("GetCameraSerialNumber returned:",ret,"Serial No:",iSerialNumber)
  
  #Clean up
  (ret) = sdk.ShutDown()
  print("Shutdown returned",ret)

else:
  print("Cannot continue, could not initialise camera")
  
