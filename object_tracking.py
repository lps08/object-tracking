import imutils
from imutils.video import VideoStream, FPS
import cv2

class ObjectTracking(object):
    """Object tracking is one such application of computer vision 
    where an object is detected in a video, otherwise interpreted 
    as a set of frames, and the object’s trajectory is estimated.
    
    Parameters
    ----------
    - src: video source. If you want to use the web cam, set as '0',  
    otherwise set the video path. Defaut = 0.

    - tracker_algorithm: the tracker algorithm can be:
        - csrt
        - kcf
        - boosting
        - mil
        - tld
        - medianflow
        - mosse
    """

    def __init__(self, src:str="0", tracker_algorithm:str="csrt") -> None:
        self.src = src
        self.tracker_algorithm = tracker_algorithm
        self.boundingbox = None
        self.fps = FPS()
        self.video = self.get_video()

    def run(self):
        """This function runs the main loop to reprocuces the video with 
        tracker algorithm.
        """
        tracker = self.get_tracker()

        while True:
            frame = self.video.read() if self.src is "0" else self.video.read()[1]

            if frame is None:
                break

            frame = imutils.resize(frame, width=500)

            self.draw_boundingbox(frame=frame, tracker=tracker, boundingbox=self.boundingbox, frame_size=frame.shape[:2])

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                self.boundingbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

                tracker.init(frame, self.boundingbox)
                self.fps.start()

            if key == ord("e"):
                self.boundingbox = None
                tracker = self.get_tracker()
            
            elif key == ord("q"):
                break

        self.video.stop()
        cv2.destroyAllWindows()

    def draw_boundingbox(self, frame, tracker, boundingbox, frame_size:tuple):
        """This function draws and update the bounding box from the tracker 
        algorithm.

        Parameters
        ----------
        - frame: the current frame that will be used to draw and update the 
        new bounding box.
        - tracker: tracker object.
        - boundingbox: bounding box selected to be tracked.
        - frame_size: video frame size.
        """
        if boundingbox is not None:
            (success, box) = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            self.fps.update()
            self.fps.stop()

            info = [
                ("Tracker", 'csrt'),
                ("Success", "Yes" if success else "No"),
                ("FPS", f"{self.fps.fps()}"),
            ]

            for (i, (k, v)) in enumerate(info):
                text = f"{k}: {v}"
                cv2.putText(frame, text, (10, frame_size[1] - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def get_tracker(self):
        """Get the tracker object selected.

        Returns
        -------
        - tracker algorithm selected
        """
        object_trackers = {
            "csrt": cv2.legacy.TrackerCSRT.create(),
            "kcf": cv2.legacy.TrackerKCF.create(),
            "boosting": cv2.legacy.TrackerBoosting.create(),
            "mil": cv2.legacy.TrackerMIL.create(),
            "tld": cv2.legacy.TrackerTLD.create(),
            "medianflow": cv2.legacy.TrackerMedianFlow.create(),
            "mosse": cv2.legacy.TrackerMOSSE.create()
        }

        return object_trackers[self.tracker_algorithm]

    def get_video(self):
        """Select the video type.

        Returns:
        - VideoStream: if the selected source video was "0".
        - VideoCapture from vídeo: if the selected source video was a video path.
        """
        if self.src is "0":
            return VideoStream(src=0).start()
        else:
            return cv2.VideoCapture(self.src)

if __name__ == "__main__":
    tracker = ObjectTracking(tracker_algorithm="csrt")
    tracker.run()