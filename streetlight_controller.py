import cv2
import numpy as np
from ultralytics import RTDETR
import time
from collections import defaultdict

class StreetLightController:
    def __init__(self,model_path, labels_path='labels.txt'):
        self.model = RTDETR(model_path).to('cuda')

        self.lamp_status = {}
        self.lamp_zones = self._define_lamp_zones()
        self.object_history = defaultdict(list)

    def load_labels(self,path):
        self.class_labels = {}
        with open(path) as f:
            for line in f:
                idx, name = line.strip().split(' ',1)
                self.class_labels[int(idx)] = name
        self.vehicle_classes = ['bike','bus','car','motor','rider','truck']

    def _define_lamp_zones(self):
        return {
            1: (0, 0, 320, 240),    
            2: (320, 0, 640, 240),   
            3: (0, 240, 320, 480),   
            4: (320, 240, 640, 480)  
        }
    
    def detect_objects(self, frame):
        results = self.model.track(frame, persist=True, tracker='bytetrack.yaml')
        return results
    
    def analyze_traffic_patterns(self,detections):
        # JM PART
        pass

    def control_lights(self, detections):
        active_lamps = set()

        for detection in detections:
            x_center = detection['x'] + detection['w']/2
            y_center = detection['y'] + detection['h']/2

            for lamp_id, zone in self.lamp_zones.items():
                x1,y1,x2,y2 = zone
                if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                    active_lamps.add(lamp_id)
                    break
        
        for lamp_id in self.lamp_zones.keys():
            self.lamp_status[lamp_id] = lamp_id in active_lamps

    def visualize(self, frame, detections):
        annotated_frame = detections[0].plot()

        for lamp_id, zone in self.lamp_zones.items():
            color = (0, 255, 0) if self.lamp_status.get(lamp_id, False) else (0, 0, 255)
            cv2.rectangle(annotated_frame, (zone[0], zone[1]), (zone[2], zone[3]), color, 2)
            cv2.putText(annotated_frame, f"Lamp {lamp_id}", (zone[0]+10, zone[1]+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated_frame
