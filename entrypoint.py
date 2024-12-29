from bag_counter_pipeline import BagCounter
from ultralytics import YOLO
import argparse


if __name__ == '__main__':
    model_path = "model/yolov8S_best.pt"
    model = YOLO(model_path)

    parser = argparse.ArgumentParser(prog='YOLO bag detector')         
    parser.add_argument('-mc', '--modelconf', default=0.5)
    parser.add_argument('-fwu', '--frame_without_updates', default=17)
    parser.add_argument('-ioutr', '--iou_treshold', default=0.75) 
    parser.add_argument('input_video_path')
    parser.add_argument('output_video_path')
    args = parser.parse_args()

    model_conf=args.modelconf
    frame_without_updates = args.frame_without_updates
    iou_treshold=args.iou_treshold

    input_video_name = args.input_video_path
    output_video_name = args.output_video_path

    obj = BagCounter(model=model,
                     model_conf=model_conf,
                     frame_without_updates=frame_without_updates,
                     iou_treshold=iou_treshold)

    obj.video_processing(input_video_path = input_video_name,
                         output_video_path = output_video_name,
                         show_video=False)
    

