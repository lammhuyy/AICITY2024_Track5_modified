import association, kalmanfilter, ocsort
import numpy as np
import json
from collections import defaultdict

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    return obj

def apply_consistent_labels(tracking_result):
    """
    Modify tracking_result in-place to ensure each object has a consistent label across all frames.
    Uses confidence-weighted voting to select the most likely label per object.
    
    Args:
        tracking_result (dict): {track_id: list of [x1, y1, x2, y2, conf, label, frame_id]}
    """
    consistent_labels = {}

    # Step 1: Determine the consistent label for each track_id
    for track_id, boxes in tracking_result.items():
        label_votes = defaultdict(float)

        for box in boxes:
            conf = box[4]
            label = int(box[5])
            label_votes[label] += conf

        # Pick the label with the highest total confidence
        consistent_label = max(label_votes.items(), key=lambda x: x[1])[0]
        consistent_labels[track_id] = consistent_label

    # Step 2: Rewrite the original dict using consistent labels
    for track_id, boxes in tracking_result.items():
        final_label = consistent_labels[track_id]
        for box in boxes:
            box[5] = final_label  # Update the label
    return tracking_result


def process_tracking_result(tracking_result):
    tracking_result = convert_ndarray_to_list(tracking_result)
    for track_id in tracking_result:
        detected_result = []
        for age, bbox in tracking_result[track_id].items():
            detected_result.append(bbox)
        tracking_result[track_id] = detected_result
    tracking_result = apply_consistent_labels(tracking_result)
    return tracking_result
            
def convert_tracking_result_to_frames(tracking_result):
    """
    Convert tracking_result to a list of frames, each containing detections for different objects.
    Each detection contains bounding box (x1, y1, x2, y2), label, score, track_id.
    
    Args:
    - tracking_result (dict): A dictionary where the keys are track_ids, and the values are lists of detections.
    
    Returns:
    - List of frames, where each frame is a list of detections.
    """
    # Find the max frame_id to know how many frames we have
    max_frame_id = max([box[6] for boxes in tracking_result.values() for box in boxes])

    # Initialize a list to hold frames
    frames = [[] for _ in range(max_frame_id)]

    # Organize detections into frames
    for track_id, boxes in tracking_result.items():
        for box in boxes:
            x1, y1, x2, y2, score, label, frame_id = box
            frame_id = int(frame_id)  # Ensure frame_id is an integer
            
            # Create detection dictionary
            detection = {
                'track_id': track_id,
                'bounding_box': [x1, y1, x2, y2],
                'label': label,
                'score': score
            }
            
            # Add the detection to the corresponding frame
            frames[frame_id - 1].append(detection)  # frame_id starts at 1, so use frame_id - 1 as index
    
    return frames

# Example: Convert tracking result for multiple videos
def process_multiple_videos_tracking_results(videos_tracking_results):
    video_frames = []

    for tracking_result in videos_tracking_results:
        frames = convert_tracking_result_to_frames(tracking_result)
        video_frames.append(frames)
    
    return video_frames

tracking_result = {}
det_results = [
    {  # Frame 1
        "bboxes": [[120, 230, 540, 900], [320, 150, 760, 840], [180, 400, 620, 920], 
                   [200, 300, 710, 880], [150, 220, 600, 910], [90, 270, 550, 870]],
        "scores": [0.89, 0.76, 0.91, 0.67, 0.82, 0.74],
        "labels": [1, 2, 3, 4, 5, 6]
    },
    {  # Frame 2 - Object 6 disappears, new object 7 appears
        "bboxes": [[125, 235, 545, 905], [325, 155, 765, 845], [185, 405, 625, 925], 
                   [205, 305, 715, 885], [155, 225, 605, 915], [250, 360, 700, 890]],
        "scores": [0.90, 0.77, 0.92, 0.68, 0.83, 0.75],
        "labels": [1, 2, 3, 4, 5, 7]
    },
    {  # Frame 3 - Object 1 disappears, Object 6 reappears
        "bboxes": [[330, 160, 770, 850], [190, 410, 630, 930], [210, 310, 720, 890], 
                   [160, 230, 610, 920], [260, 370, 710, 900], [95, 275, 555, 860]],
        "scores": [0.78, 0.93, 0.69, 0.84, 0.76, 0.74],
        "labels": [2, 3, 4, 5, 7, 6]
    },
    {  # Frame 4 - Objects 3 and 4 disappear, Object 8 appears
        "bboxes": [[335, 165, 775, 855], [165, 235, 615, 925], [270, 380, 720, 910], 
                   [100, 280, 560, 870], [300, 170, 750, 880]],
        "scores": [0.79, 0.85, 0.77, 0.75, 0.80],
        "labels": [2, 5, 7, 6, 8]
    },
    {  # Frame 5 - Objects 1 and 3 reappear, Objects 2 and 5 disappear
        "bboxes": [[130, 240, 550, 910], [185, 405, 625, 930], [275, 385, 725, 915], 
                   [105, 285, 565, 875], [310, 175, 755, 885]],
        "scores": [0.91, 0.94, 0.78, 0.76, 0.82],
        "labels": [1, 3, 7, 6, 8]
    }
]

tracker = ocsort.OCSort(det_thresh=0.4, iou_threshold=0.5, max_age=1, min_hits=1)

for i, det_result in enumerate(det_results):
    # final_bboxes = np.array(det_result["bboxes"], dtype=np.float32)
    # final_scores = np.array(det_result["scores"], dtype=np.float32)
    # final_labels = np.array(det_result["labels"], dtype=np.float32)
    final_bboxes = det_result["bboxes"]
    final_scores = det_result["scores"]
    final_labels = det_result["labels"]

    bbox_xyxyc = np.hstack((final_bboxes, np.c_[final_scores], np.c_[final_labels]))
    # print(bbox_xyxyc.shape)

    print("Frame", i + 1)
    tracks = tracker.update(bbox_xyxyc, (1000, 1000), (1000, 1000), i)
    print("tracks:", tracks)

    # for track in tracks:
    #     bbox = track[:4]
    #     track_id = track[4]
    #     label = track[5]
    #     conf_score = track[6]
    #     if track_id not in tracking_result:
    #         tracking_result[track_id] = []
    #     tracking_result[track_id].append([i, bbox[0], bbox[1], bbox[2], bbox[3], label, conf_score])

for obj in tracker.trackers:
    track_id = obj.id
    tracker.all_observations[track_id] = obj.observations

print("ALL OBSERVATIONS")
for id, data in tracker.all_observations.items():
    print(id)
    print(data)

with open("tracking_result.json", 'w') as f:
    serializable_data = convert_ndarray_to_list(tracker.all_observations)
    processed_tracking_result = process_tracking_result(serializable_data)
    json.dump(processed_tracking_result, f, indent=4)

with open("cons_tracking_result.json", 'w') as f:
    cons_tracking_result = apply_consistent_labels(processed_tracking_result)
    print(processed_tracking_result)
    json.dump(processed_tracking_result, f, indent = 4)