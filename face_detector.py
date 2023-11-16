from mtcnn import MTCNN
from utils import to_cuda


class FaceDetector:
    def __init__(self) -> None:
        self.detector = MTCNN()
        self.detector.eval()
        self.detector = to_cuda(self.detector)

    def get_area(self, box):
        w, h = box[2], box[3]

        return w * h

    def detect_face(self, img, person_id, confidence=0.8):
        """
        Input: img, person_id
        Output: { cropped_face, person_id }
        """
        confidence = 0.9
        detected_boxes = self.detector.detect_faces(img)

        detected_boxes.sort(key=lambda x: self.get_area(x["box"]), reverse=True)

        result = {"image": None, "person_id": person_id}

        for dect in detected_boxes:
            if dect["confidence"] >= confidence:
                x, y, w, h = dect["box"]
                cropped_face = img[y : y + h, x : x + w].copy()
                result["image"] = cropped_face
                break

        return result
