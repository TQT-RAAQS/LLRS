# Low-Latency Control System Summary

The Low-Latency Control System is the system responsible for running the LLRS experiment shots. The LLCS connects to the workstation via a ZMQ server and communicates through the server to configure hardware, program the experimental shot sequence, and to return relevant experimental metadata. 

# General Workflow:

<p align="center">
  <img src="llcs-design.png" width="350" title="LLCS Workflow Diagram">
</p>


```BEGIN```: Connects to hardware, configures LLRS with initial configuration, and streams static waveforms

```IDLE```: Waits for server requests

```RESET```: Reconfigures LLRS, returns to IDLE upon completion

```PSF_RESET```: Updates PSF file, returns to IDLE upon completion

```Close AWG```: LLCS gives up control of the AWG

```Restart AWG```: LLCS retakes control of the AWG, returns to IDLE upon completion

```CONFIG```: Configures experiment (hardware and experimental shot sequence), transitions to READY upon completion

```READY```: Awaits hardware trigger to begin the configured experiment sequence

```Experiment Sequence```: Executes the series of experiment modules. Transitions to DONE upon finishing the current experiment trigger
 
```DONE```: Saves relevant metadata to a JSON file. If there are more triggers remaining in the shot, transition to READY, otherwise transition to IDLE.

```EXIT```: Exits the LLCS


# Usage Instructions:

### 1. Navigate to the low-latency-LLCS directory. 
```Bash
cd ~/Experiment/experiment/modules/low-latency-LLCS
```

### 2. Set the appropiate Shell Variables:
```Bash
export CPATH="$CPATH:/usr/include/hdf5/serial/:/usr/include/hdf5"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/Experiment/experiment/modules/low-latency-LLCS/lib:/home/$USER/Experiment/experiment/instruments/awg_cpp/lib:/home/$USER/Experiment/experiment/instruments/fgc_cpp/lib"
```

### 3. Run the python script to compile all relevant modules
```Bash
./build.py
```
If there is an error message regarding denied permission, simply add the execution permission to the file, and run the script again.
```Bash
chmod a+x build.py
```

### 4. Run the LLCS
```Bash
./bin/main
```

### 5. Wait for the message "LLCS:: IDLE state". In the IDLE state you can send 5 server requests.

 ```*.h5```<p></p>
        - The request string can be an h5 filepath containing the configuration data <p></p>
        - Configures the AWG, "easy to configure" properties of the LLRS, and the experimental shot sequence<p></p>
        - Automatically transitions to the READY state upon completion

 ```RESET```<p></p>
        - Closes connection to the LLCS for another device to take over<p></p>
        - Waits for the "control" request from the server to retake control over the AWG<p></p>
        - IMPORTANT: The user must follow up with sending "control" before proceeding with any other LLCS actions<p></p>
        - Automatically transition back to IDLE

  ```psf```<p></p>
        - Runs the psf_translator python script, which updates psf file<p></p>
        - Automatically transitions back to the IDLE state<p></p>
        - IMPORTANT: The user should send a .h5 filepath when the LLCS is back in IDLE in order to update the LLRS with the new psf file

 ```{ *```<p></p>
        - The request string is the contents of a JSON file containing the LLRS configuration file, beginning with "{"<p></p>
        - Reconfigures the LLRS and it's waveform table with the new configuration<p></p>
        - This will take a few minutes to complete, and will automatically transition to IDLE upon completion

```send_data```<p></p>
        - Return a filepath to the workstation for the metadata file of the most recent shot
    
 ```done```<p></p>
        - The LLCS program will terminate gracefully



### 6. After sending a .h5 filepath while in the IDLE state, the LLCS will transition to READY. Wait for the message "Awaiting Hardware Trigger...". The user must send a hardware trigger via the workstation to run each experiment sequence. This message will reappear every time there are more triggers left in the experimental shot. 



### 7. After the experimental shot is completed, the LLCS will transition back to IDLE. Repeat steps 4-5.

