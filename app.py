

from time import time
from flask import Flask
import cv2,os

from flask import jsonify 
from skimage import io

from flask_cors import CORS, cross_origin

from yolodetect import YoloDetect
from flask import request
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "C:/Users/PCLTC/Desktop/yolov7/static"

classes = ['person'] #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

opt  = {
    
    "weights": "C:/Users/PCLTC/Desktop/yolov7/models/yolov7-tiny.pt", # Path to weights file default weights are for nano model
    "yaml"   : "C:/Users/PCLTC/Desktop/yolov7/data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.6, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes  # list of classes to filter or None
}
model = YoloDetect(opt=opt)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
@app.route('/v1/vision/detection', methods=['POST'] )
def predict():
    image = request.files['image']
    if image:
        # Lưu file
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        print("Save = ", path_to_save)
        image.save(path_to_save)
        # url= "https://nhatran.duckdns.org/local/time.png?v=2"
        frame = cv2.imread(path_to_save)
        # Nhận diên qua model Yolov7
        frame, num_object,predictions = model.detect(frame);
        # if num_object >0:
        #     cv2.imwrite(path_to_save, frame)
        # print(predictions)
        del frame
        # Trả về đường dẫn tới file ảnh đã bounding box
        return jsonify(
            success = True,
            predictions = predictions
        )

    return jsonify(
        success = False
    )


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=81)