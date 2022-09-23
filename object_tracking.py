import imutils
from imutils.video import VideoStream, FPS
import cv2

class ObjectTracking(object):
    def __init__(self, tracker_algorithm:str="csrt") -> None:
        self.tracker_algorithm = tracker_algorithm
        self.boundingbox = None
        self.vs = VideoStream(src=0).start()
        self.fps = FPS()

    def run(self):
        tracker = self.get_tracker()

        while True:
            frame = self.vs.read()

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

        self.vs.stop()
        cv2.destroyAllWindows()

    def draw_boundingbox(self, frame, tracker, boundingbox, frame_size:tuple):
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

if __name__ == "__main__":
    tracker = ObjectTracking(tracker_algorithm="mosse")
    tracker.run()