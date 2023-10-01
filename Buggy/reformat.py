from PIL import Image
import glob
import os, os.path

image_list = []

for f in os.listdir("ArTaxOr/"):
    info = os.path.splitext(f)
    if not len(info[1]):
        for file in glob.glob(f"ArTaxOr/{info[0]}/*.jpg"):
            print(file)
            file_path = file.split("/")
            im = Image.open(file)
            im = im.resize((600,600))
            im = im.save(f"Data/{info[0]}/{file_path[len(file_path) - 1]}")
