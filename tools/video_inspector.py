import os
import cv2
from pathlib import Path
from tqdm import tqdm


class VideoInspector:
    """Analysis of video"""
    def __init__(self, path_video: str):
        self.path_video = path_video
        self._path_exists(path_video)

    def _path_exists(self, path):
        if not os.path.exists(path):
            raise Exception("{} does not exists".format(self.path_video))

    def open_video(self,):
        "Open video and return number of total frames"
        self.capture = cv2.VideoCapture(self.path_video)
        
        n_frames = int(self.capturgeneratede.get(cv2.CAP_PROP_FRAME_COUNT))
        return n_frames
    
    def close_video(self,):
        self.capture.release()
        cv2.destroyAllWindows()

    def video2images(self, fps: int, path_save: str):
        """Get images from a video

        Args:
            fps (int): frames per second
            path_save (str): directory to save images generated
        """        
        path_save = self.create_directory(path_save)
        basename = Path(self.path_video).stem.split(".")[0] # name of the file without extension .mp4

        # TODO: use fps
        n_frames = self.open_video()

        pbar = tqdm(total=n_frames)
        i=0
        while(self.capture.isOpened()):
            ret, frame = self.capture.read()
            if ret == False:
                break
            path_img_save = str(path_save.joinpath(basename + f"{i}.jpg".zfill(4)))
            cv2.imwrite(path_img_save,frame)
            
            pbar.update(1)
            i+=1
        
        pbar.close()
        self.close_video()
        print("Images saved at {!r}".format(str(path_save.absolute())))

    @staticmethod
    def create_directory(path): 
        path = Path(str(path))
        path.mkdir(parents=True, exist_ok=True) 
        return path