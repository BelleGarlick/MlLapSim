import os
import shutil



output_paths = '/Users/belle/Downloads/Version 3/'

convertion_dirs = [
    output_paths + "training",
    output_paths + "test",
    output_paths + "validation",
    output_paths + "faulty",
]



def open_path(f):
    with open(f, "r") as file:
        return file.read()


def convert_file(f):
    data = open_path(f).split("\n")
    data = data[0:1] + data[1::2]
    with open(f, "w") as file:
        file.write("\n".join(data))


def scan_files(dir):
    files = os.listdir(dir)
    files = [f for f in files if not f.startswith(".")]

    for i, file in enumerate(files):
        if file.startswith("."): continue

        print(f"\r{i+1}/{len(files)} {file}", end="")

        path_dir = os.path.join(dir, file)
        convert_file(path_dir + "/track.csv")
        convert_file(path_dir + "/boundaries.csv")

        # breakpoint()


if __name__ == "__main__":
    for convertion_dir in convertion_dirs:
        scan_files(convertion_dir)