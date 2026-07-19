import os
import tarfile
import urllib.request

URL = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"

SAVE_DIR = "data"
ARCHIVE = os.path.join(
    SAVE_DIR,
    "ibug_300W_large_face_landmark_dataset.tar.gz"
)

EXTRACT_DIR = os.path.join(
    SAVE_DIR,
    "ibug_300W_large_face_landmark_dataset"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# Download
print("Downloading dataset...")
urllib.request.urlretrieve(URL, ARCHIVE)
print("Download complete!")

# Verify archive
print("Checking archive...")
if not tarfile.is_tarfile(ARCHIVE):
    raise Exception("Downloaded archive is corrupted.")

print("Archive verified.")

# Extract
print("Extracting...")

with tarfile.open(ARCHIVE, "r:gz") as tar:
    tar.extractall(path=SAVE_DIR)

print("Done!")