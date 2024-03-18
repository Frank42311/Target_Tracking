import os
import cv2
import pydicom
import numpy as np

# Path to the root directory of DICOM files.
# This could be 'Moving' or 'G:/Moving', depending on your specific setup.
root_dir: str = 'Moving'

# Video output settings
fps: int = 24  # Frame rate
video_size: tuple[int, int] = (256, 256)  # Adjust based on the size of your DICOM images
video_output: str = 'output_video.avi'

# Initialize the video writer
# The four-character code used to specify the video codec. 'XVID' is one such codec.
fourcc: int = cv2.VideoWriter_fourcc(*'XVID')
# VideoWriter object that we'll use to write the video. Parameters include the output file name,
# the codec, frames per second, and video size.
video_writer: cv2.VideoWriter = cv2.VideoWriter(video_output, fourcc, fps, video_size)

# Iterate over the DICOM files
for subfolder in sorted(os.listdir(root_dir)):
    dicom_folder: str = os.path.join(root_dir, subfolder)
    # Ensure we're processing directories only
    if not os.path.isdir(dicom_folder):
        continue
    for dicom_file in sorted(os.listdir(dicom_folder)):
        # Process only .dcm files
        if not dicom_file.endswith('.dcm'):
            continue

        dicom_path: str = os.path.join(dicom_folder, dicom_file)
        # Read the DICOM file
        dataset: pydicom.dataset.FileDataset = pydicom.dcmread(dicom_path)

        # Convert the DICOM image to an 8-bit grayscale image
        image: np.ndarray = dataset.pixel_array
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)

        # Convert to a 3-channel grayscale image to match the video writer's requirements
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Resize the image to fit the video size
        image_resized: np.ndarray = cv2.resize(image, video_size)

        # Write the image to the video
        video_writer.write(image_resized)

# Release the video writer
video_writer.release()

print("Video creation completed!")
