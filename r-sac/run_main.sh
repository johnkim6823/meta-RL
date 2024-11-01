#!/bin/bash

# Define the log file name with a timestamp
LOG_FILE="main_$(date +'%Y%m%d_%H%M%S').log"

# Run the Python script in the background and redirect stdout and stderr to the log file
nohup python main.py > "$LOG_FILE" 2>&1 &

# Get the process ID (PID) of the last background command
PID=$!

# Print a message indicating where the log has been saved and the PID of the background process
echo "Script is running in the background with PID $PID. Log saved to $LOG_FILE"
