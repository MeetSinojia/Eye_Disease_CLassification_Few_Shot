from imutils import paths
import argparse
import time
import sys
import cv2
import os

def dhash(image, hashSize=8):
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
	help="dataset of images to search through (i.e., the haytack)")
args = vars(ap.parse_args())

print("[INFO] computing hashes for haystack...")
haystackPaths = list(paths.list_images(args["haystack"]))

if sys.platform != "win32":
	haystackPaths = [p.replace("\\", "") for p in haystackPaths]
map_image_to_hash = {}
for p in haystackPaths:
	image = cv2.imread(p)

	if image is None:
		continue
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imageHash = dhash(image)
	l = map_image_to_hash.get(imageHash, [])
	l.append(p)
	map_image_to_hash[imageHash] = l

duplicates_dict = {}
for k, v in map_image_to_hash.items():
    if len(map_image_to_hash[k]) > 1:
        duplicates_dict[k] = v

print(duplicates_dict)

for k, v in duplicates_dict.items():
    for filepath in v[1:]:
        os.remove(filepath)