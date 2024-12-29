# YOLO detection

This code detects bags on a moving conveyor.

To run, you need to have docker installed

Run the following commands:
### build docker image
```bash
docker build . -t box_detection
```
### run docker container
```bash
docker run --volume {local_path}:/app/data box_detection /app/data/input.mp4 /app/data/output.mp4 --modelconf 0.5 --frame_without_updates 17 --iou_treshold 0.75
```
Command line arguments:
- input_video_path - path to input video inside docker container
- output_video_path - path to output video inside docker container
- modelconf - threshold for model confidence score
- frame_without_updates - the number of frames to wait to forget about detected boxes
- iou_treshold - the IoU threshold used to determine if a detected object matches an existing tracked object 
