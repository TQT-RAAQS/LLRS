# Andor iXon Ultra 888

Performs all utility from the Andor SDK

`ret` == `20002` signifies successful operation

To configure camera to send images via Camera Link be sure to set `cameralink_mode` in EmccdMetaInformation = `1`

### Init
Must pass in instance of EmccdMetaInformation

```python
metaInformation = EmccdMetaInformation(...)
andoriXonUltra888 = AndoriXonUltra888(metaInformation)
```

### set_static_properties()
Sets properties that _should not_ vary from acquisition to acquisiton provided in meta information. 
```python
stabalized_temp = andoriXonUltra888.cool_sensor()
```

### set_acquisition_properties()
Sets properties that _should_ vary from acquisition to acquisiton provided in meta information. 
```python
andoriXonUltra888.set_acquisition_properties()
```

### cool_sensor()
Will cool sensor to temperature provided in meta information. Blocks until stabalized.
```python
andoriXonUltra888.set_acquisition_properties()
```

### acquire_images()
Starts acquisition given preset parameters. Stores images to self.image_buffer. 
```python
andoriXonUltra888.acquire_images()
```

### save_images()
Saves image buffer and meta information to a file location specified in arguments
```python
andoriXonUltra888.save_images(folder, file_name)
```

### shutdown_camera()
Safely turns off SDK and cooler
```python
andoriXonUltra888.shutdown_camera()
```

### abort_acquisition()
Aborts the current acquisition and returns the index of the most recent image to retrieve the image buffer with
```python
image_index = andoriXonUltra888.abort_acquisition()
```

### get_image_buffer()
Retrieves the image buffer of the most recent acquisition
```python
image_buffer = andoriXonUltra888.get_image_buffer()
```

### get_fastest_vs_speed()
Retrieves recommended fastest vs speed from the SDK
```python
andoriXonUltra888.get_fastest_vs_speed()
```

### get_all_hs_speeds()
Retrieves all horizontal shift speeds from the SDK
```python
andoriXonUltra888.get_all_hs_speeds()
```

### get_all_vs_speeds()
Retrieves all vertical shift speeds from the SDK
```python
andoriXonUltra888.get_all_vs_speeds()
```

### get_minimum_achievable_temperature()
Retrieves all vertical shift speeds from the SDK
```python
andoriXonUltra888.get_all_vs_speeds()
```