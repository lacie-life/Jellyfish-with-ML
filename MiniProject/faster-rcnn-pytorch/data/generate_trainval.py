
import sys
import os

heridal_path = "/home/jun/Data/heridal/trainImages/labels"

def generate(ann_folder):
    files = sorted([x.replace(".xml", "") for x in os.listdir(ann_folder) if ".xml" in x])
    with open(os.path.join(ann_folder, "trainval.txt"), "w") as out:
        for f in files:

            out.write(f + "\n")
    print("Generated trainval.txt in", ann_folder)

if __name__ == "__main__":
    # if not sys.argv[1:]:
    #     print("Usage: python generate_trainval.py [pascal_voc_annotations_folder]")
    #     sys.exit(-1)
    generate(heridal_path)
    sys.exit(0)