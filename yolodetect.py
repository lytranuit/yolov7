from pickle import FALSE
from tkinter import Frame
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
import datetime
import threading
import torch
import random
from models.experimental import attempt_load
from utils.torch_utils import select_device,TracedModel

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    print(polygon.contains(centroid))
    return polygon.contains(centroid)


classes_to_filter = None #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

opt  = {
    
    "weights": "models/yolov7-tiny.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.6, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None
}

class YoloDetect():
    def __init__(self, opt=opt):
        # Parameters
       
        
        set_logging()
        weights, imgsz = opt['weights'], opt['img-size']
        self.device = select_device(opt['device'])
        half = self.device.type != 'cpu'
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))

        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:
                classes.append(names.index(class_name))

        if classes:
            classes = [i for i in range(len(names)) if i in classes]
        # Load model
        self.classes = classes
        self.colors = colors
        self.model = model
        self.stride = stride
        self.imgsz = imgsz
        self.half = half
        self.names = names
        
        # self.get_output_layers()
        self.last_alert = None
        self.alert_telegram_each = 15  # seconds
        

    def read_class_file(self):
        with open(self.classnames_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def get_output_layers(self):
        layer_names = self.model.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]

    def draw_prediction(self, img, class_id, x, y, x_plus_w, y_plus_h, points):
        label = str(self.classes[class_id])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Tinh toan centroid
        centroid = ((x + x_plus_w) // 2, (y + y_plus_h) // 2)
        cv2.circle(img, centroid, 5, (color), -1)

        if isInside(points, centroid):
            img = self.alert(img)

        return isInside(points, centroid)

    def alert(self, img):
        cv2.putText(img, "ALARM!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # New thread to send telegram after 15 seconds
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
            self.last_alert = datetime.datetime.utcnow()
            cv2.imwrite("alert.png", cv2.resize(img, dsize=None, fx=0.2, fy=0.2))
            # thread = threading.Thread(target=send_telegram)
            # thread.start()
        return img

    def detect(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment= False)[0]
        predictions = [];
        # print(self.classes)
        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= self.classes, agnostic= False)
        t2 = time_synchronized()
        n = 0; # number of detect
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
        
                for *xyxy, conf, cls in reversed(det):
                    # label = f'{self.names[int(cls)]} {conf:.2f}'
                    
                    # plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    
                    label = self.names[int(cls)]
                    confidence = int(conf.item() * 100)
                    x_min = int(xyxy[0])
                    x_max = int(xyxy[2])
                    y_min = int(xyxy[1])
                    y_max = int(xyxy[3])
                    predictions.append({"label": label,"confidence":confidence,"x_max":x_max,"x_min":x_min,"y_max":y_max,"y_min":y_min})
                    
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')        
        return img0,n,predictions

