"""
Generate images from videos
"""
from pathlib import Path
from tools import VideoInspector

# Find all videos
list_videos = [str(video) for video in Path("/mnt/atrys/Desafios/geof/videos").rglob("*.mp4")]

PATH_SAVE = 'images-from-video'
for video in list_videos:

    # Instantiate VideoInspector with one video
    inspector = VideoInspector(video)

    # Get images
    inspector.video2images(5, PATH_SAVE)