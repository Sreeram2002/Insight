import subprocess
import os
import signal

# Dictionary to map options to file paths
files = {
    '1': '/home/itsbighero6/insight/runcamera.sh',
    '2': '/home/itsbighero6/bash.sh',
    '3': '/home/itsbighero6/text_detector_using-EAST-master/Insight.py'
}

# Store the currently running process
current_process = None

def execute_file(file_path):
    """Execute the specified file."""
    global current_process

    # Terminate the current process if it exists
    if current_process:
        current_process.terminate()

    # Check if the file is a Python script
    if file_path.endswith('.py'):
        current_process = subprocess.Popen(['/usr/bin/python3', file_path])
    else:
        current_process = subprocess.Popen([file_path])

def main():
    while True:
        # Display options to the user
        print("Select a file to execute:")
        print("1. Obstacle Navigation")
        print("2. Object Recognition")
        print("3. Text to Speech")

        # Get user input
        choice = input("Enter your choice (1/2/3): ")

        # Execute the chosen file based on user input
        if choice in files:
            file_path = files[choice]
            execute_file(file_path)
        else:
            print("Invalid choice. Please select a valid option (1/2/3).")

if __name__ == "__main__":
    main()

