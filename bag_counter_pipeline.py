import cv2
import numpy as np


class BagCounter:
    """ 
    A class used to count and track bags in video frames 
    using object detection models. 
    
    Attributes:
    model (object YOLO): The object detection model to be used for bag detection 
    model_conf (float): The confidence threshold for the model's predictions 
    frame_without_updates (int): The number of frames to wait before forget 
                                 about detected boxes
    iou_treshold (float): The IoU threshold used to determine if a detected object 
                          matches an existing tracked object 
    """

    def __init__(self,
                 model,
                 model_conf:float,
                 frame_without_updates:float,
                 iou_treshold:float):
        """
        Initializes the BagCounter with the specified model, confidence threshold, 
        frame update interval, and IoU threshold
        """
        self.model = model

        self.conf = model_conf
        self.no_updates = frame_without_updates
        self.IoU_treshold = iou_treshold

        self.current_bboxes = []
        self.current_shot = 0
        self.tracker = 0
        
    def yolo_predict(self, 
                     shot:np.ndarray):
        """ 
        Runs the YOLO model on a given frame to detect objects. 
        
        Parameters: 
        shot (np array): The image on which object detection needs to be performed. 
        
        Returns:
        results (YOLO object): The detection results containing bounding boxes, 
                               confidence scores, and class labels. 
        """
        results = self.model.predict(shot, conf=self.conf)
        return results

    @staticmethod
    def get_IoU(box1:tuple, 
                box2:tuple) -> float:
        
        """ 
        Calculates the Intersection over Union (IoU) between two bounding boxes 
        
        Parameters:
        box1 (tuple): bbox coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        box2 (tuple): bbox coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        
        Returns:
        iou (float) The IoU value between the two bounding boxes 
        """

        x1_1, y1_1, x2_1, y2_1 = box1 
        x1_2, y1_2, x2_2, y2_2 = box2 

        x_left = min(x1_1, x1_2) 
        y_top = min(y1_1, y1_2) 
        x_right = max(x2_1, x2_2) 
        y_bottom = max(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top: return 0

        intersection_area = (x_right - x_left) * (y_bottom - y_top) 
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1) 
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2) 
        
        union_area = box1_area + box2_area - intersection_area 
        iou = intersection_area / union_area
        return iou

    def add_new_object(self, 
                       bbox:tuple) -> dict:
        """ 
        Adds a new detected object to the current list of detected boxes 

        Parameters: 
        bbox (tuple): bbox coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

        Returns:
        the_last_added_box (dict): the last added box with keys
        """

        pattern = {'number':self.tracker,
                   'bbox' : bbox,
                   'current_shot' : self.current_shot}
        self.current_bboxes.append(pattern)
        the_last_added_box = self.current_bboxes[-1]
        return the_last_added_box
    
    def update(self, 
               detected_box:tuple) -> dict:
        """ 
        Updates the list of tracked objects with a new detected bounding box 

        Parameters:
        detected_box (tuple): detected box (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

        Returns:
        dict: the last added box with keys 
        """

        if self.tracker == 0:
            self.tracker += 1
            return self.add_new_object(detected_box) 
        
        list_iou = [self.get_IoU(previous_value['bbox'], detected_box) 
                    for previous_value in self.current_bboxes]
        max_iou = max(list_iou)
        if max_iou > self.IoU_treshold:
            index = list_iou.index(max_iou)
            self.current_bboxes[index]['bbox'] = detected_box
            self.current_bboxes[index]['current_shot'] = self.current_shot
            return self.current_bboxes[index]
        else:
            self.tracker += 1
            return self.add_new_object(detected_box)

    def delete_useless_boxes(self):
        """
        Delete boxes that was inwalid to detect and if box doesn't update during 
        self.no_updates we delete this inwalid useless boxes
        """

        self.current_bboxes = [i for i in self.current_bboxes \
                               if self.current_shot - i['current_shot'] <= self.no_updates]

    def draw(self, 
             img: np.ndarray, 
             box: tuple,
             number: int) -> np.ndarray:
        """
        Draw detected box and number on image

        Parameters:
        img (np.ndarray): image shot 
        box (tuple): box that need to draw on image
        number (int):number of the box

        Returns:
        img (np.ndarray): image with boxes and numbers
        """
        x1_1, y1_1, x2_1, y2_1 = box
        cv2.rectangle(img, (int(x1_1), int(y1_1)),
                                (int(x2_1), int(y2_1)), (0, 0, 255), 2)
        cv2.putText(img, f"bag {number}",
                                (int(x1_1), int(y1_1) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        return img

    def detect_boxes(self,
                     img: np.ndarray) -> np.ndarray:
        """
        Detects objects in a given image and updates the tracking information. 
        
        Parameters: 
        img (np.ndarray): image that need to detect. 
        
        Returns:
        img (np.ndarray): image with detected boxes and numbers
        """

        results = self.yolo_predict(img)
        boxes_in_frame = []
        for result in results:
            for box in result.boxes:
                box = box.xyxy[0]
                boxes_in_frame.append(self.update(box))
                self.delete_useless_boxes()
        for box_info in boxes_in_frame:
            box = box_info['bbox']
            number = box_info['number']
            img = self.draw(img, box, number)
                
        return img

    def video_processing(self,
                         input_video_path='data/test_video.mp4',
                         output_video_path='data/output_video.mp4',
                         show_video=False):
        """
        Video processing and box detection

        Parameters:
        input_video_path (str): path to video for prediction
        output_video_path (str): path to save video with detected boxes
        show_video (bool): flag to show video
        """
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if cap.isOpened(): 
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        vidwriter = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        while True:
            success, img = cap.read()
            if not success:
                break
            result_img = self.detect_boxes(img)
            vidwriter.write(result_img)
            if show_video:
                cv2.imshow("Video", result_img)
                cv2.waitKey(1)
            self.current_shot += 1
        vidwriter.release()

