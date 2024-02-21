import pyzed as zed
import numpy as np
import sounddevice as sd  # Sound library for audio output

# Camera initialization
init_params = zed.InitParameters()
init_params.camera_resolution = zed.RESOLUTION.HD720  # Adjust resolution as needed
init_params.depth_mode = zed.DEPTH_MODE.PERFORMANCE  # Adjust depth mode

zed_camera = zed.Camera()

if not zed_camera.open(init_params):
    print("Failed to open Zed camera")
    exit(1)

# Define vibration intensity scale based on depth (adjust values as needed)
min_depth = 500  # mm (minimum detectable depth)
max_depth = 2000  # mm (maximum desired depth)
vibration_scale = 100  # Higher value means stronger vibration

while True:
    if zed_camera.grab():
        # Retrieve depth map
        depth_map = zed_camera.retrieve_measure(zed.MEASURE.DEPTH)
        depth_data = depth_map.get_data()

        # Find average depth (replace with specific object selection if needed)
        average_depth = np.mean(depth_data)

        # Calculate vibration intensity based on depth and scale
        vibration_intensity = int(
            ((max_depth - average_depth) / (max_depth - min_depth)) * vibration_scale
        )

        # Generate vibration sound data (adjust frequency and duration as needed)
        duration = 0.1  # seconds
        frequency = 100 + vibration_intensity
        vibration_sound = np.sin(np.linspace(0, 2 * np.pi * frequency * duration, int(duration * 44100)))

        # Play vibration sound through speakers
        sd.play(vibration_sound, samplerate=44100)
        sd.wait()

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed_camera.close()
