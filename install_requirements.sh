#!/bin/bash

FILE="requirements.txt"

# Check if requirements.txt exists
if [ -f "$FILE" ]; then
    echo "Found $FILE"
    
    # Using pip3 to ensure python3 usage; change to 'pip' if needed
    pip3 install -r "$FILE"    
    if [ $? -eq 0 ]; then
        echo "Installation complete!"
    else
        echo "An error occurred during installation."
    fi
else
    echo "Error: $FILE not found"
fi
