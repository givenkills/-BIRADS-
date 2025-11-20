from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n-cls.yaml") # build a new model from YAML
model = YOLO("yolov8x-cls.pt") # load a pretrained model (recommended for training)
#model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt") # build from YAML and transfer weights

# Train the model
#results = model.train(data=r"C:\Users\ZhiYi\OneDrive\Desktop\ultralytics-main\datasets\cancer",workers=0, epochs=100, imgsz=64)
results = model.train(data=r"C:\Users\ZhiYi\OneDrive\Desktop\photo-improve ",workers=0, epochs=100, imgsz=64)