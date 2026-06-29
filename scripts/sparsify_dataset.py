import os
import shutil

output_paths = '/Users/belle/Downloads/evaluate tracks/V2 Output/'

convertion_dirs = [
    # output_paths + "ArtificialTraining",
    # output_paths + "Artificial Test",
    # output_paths + "Artificial Validation",
    # output_paths + "Real Test",
]



def open_path(f):
    with open(f, "r") as file:
        return file.read()


def convert_file(f):
    data = open_path(f).split("\n")[::2]
    with open(f, "w") as file:
        file.write("\n".join(data))


def scan_files(dir):
    for i, file in enumerate(os.listdir(dir)):
        if file.startswith("."): continue

        print(f"{i}{file}")

        # path_dir = os.path.join(dir, file)
        # convert_file(path_dir + "/track.csv")
        # convert_file(path_dir + "/boundaries.csv")

        # breakpoint()


if __name__ == "__main__":
    for convertion_dir in convertion_dirs:
        scan_files(convertion_dir)