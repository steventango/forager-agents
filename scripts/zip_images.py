import argparse
import glob
import zipfile
from pathlib import Path


def zip_images(pattern, output_zip):
    files = glob.glob(pattern)
    with zipfile.ZipFile(output_zip, 'w') as f:
        for file in files:
            path = Path(file)
            parts = path.parts
            print(parts)
            new_name = f"{parts[-4]}-{parts[-3]}-{parts[-2]}-{parts[-1]}"
            f.write(file, new_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip video files matching a pattern.")
    parser.add_argument("pattern", type=str, nargs='?', default="results/env-figures/*/*/0/0.png", help="The glob pattern to match video files.")
    parser.add_argument("output_zip", type=str, nargs='?', default="images.zip", help="The name of the output zip file.")
    args = parser.parse_args()

    zip_images(args.pattern, args.output_zip)
