from ultralytics import YOLO

class ANPR_Cropper:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def crop_license_plate(self, image):
        results = self.model(image)
        boxes = results[0].boxes
        if len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            cropped_plate = image[y1:y2, x1:x2]
            return cropped_plate
        else:
            return None