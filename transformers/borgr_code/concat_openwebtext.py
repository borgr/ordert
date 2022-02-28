import json
import os

root, dirs, filenames = next(os.walk("/cs/labs/daphna/guy.hacohen/borgr/data/subopenwebtext"))
length = 1000000
dirname = "/cs/labs/daphna/guy.hacohen/borgr/data/"
out_path = os.path.join(dirname, f"subopenwtxt.txt")
print("Writing to out_path")
with open(out_path, "w")as out_fl:
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print(f"Dealt with {i} files")
        if i == length:
            print(f"Done with predefined number of files {length}")
            break
        with open(filename) as fl:
            out_fl.write(fl.read())

if i == length:
    restricted_out_path = os.path.join(dirname, f"{length}subopenwtxt.txt")
    os.rename(out_path, restricted_out_path)

print(f"Finished combining {i} files")