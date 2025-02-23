import argparse
import glob
import zipfile
from pathlib import Path


def zip_videos(pattern, output_zip):
    video_files = glob.glob(pattern)
    with zipfile.ZipFile(output_zip, 'w') as f:
        for video in video_files:
            video_path = Path(video)
            parts = video_path.parts
            print(parts)
            new_name = f"{parts[-4]}-{parts[-3]}-{parts[-2]}-{parts[-1]}"
            f.write(video, new_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip video files matching a pattern.")
    parser.add_argument("pattern", type=str, nargs='?', default="results/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-*/0/499000-499999.mp4", help="The glob pattern to match video files.")
    parser.add_argument("output_zip", type=str, nargs='?', default="videos.zip", help="The name of the output zip file.")
    args = parser.parse_args()

    zip_videos(args.pattern, args.output_zip)
