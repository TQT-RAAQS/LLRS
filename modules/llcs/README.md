# Low Latency Control System

The Low-Latency Control System is the system responsible for running the LLRS experiment shots. The LLCS connects to the workstation via a ZMQ server and communicates through the server to configure hardware, program the experimental shot sequence, and to return relevant experimental metadata. 

## Usage Instructions:

1. Run the LLCS place at `bin/modules/llcs/llcs`.

2. Wait for the message "LLCS:: IDLE state". In the IDLE state you can send your desired server requests.

3. After sending a .h5 filepath while in the IDLE state, the LLCS will transition to READY. Wait for the message "Awaiting Hardware Trigger...". The user must send a hardware trigger to the AWG to run each experiment sequence. This message will reappear every time there are more triggers left in the experimental shot. 

4. After the experimental shot is completed, the LLCS will transition back to IDLE. Repeat steps 4-5.
