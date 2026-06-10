import math
import os
import random
import shutil

if __name__ == "__main__":
    path = "/Users/belle/Developer/Final/Real Test"
    files = os.listdir(path)

    for file in files:
        if file.startswith("."):
            os.remove(os.path.join(path, file))
            continue

        subdirs = os.listdir(os.path.join(path, file))
        if len(subdirs) != 10:
            print(file)
            # pass

            shutil.rmtree(os.path.join(path, file))
            # subdir = os.path.join(path, subdirs[0])
        else:
            pass
            # breakpoint()

