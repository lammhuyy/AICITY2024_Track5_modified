import association, kalmanfilter, ocsort
import numpy as np

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

    print("Input", i)
    tracks = tracker.update(bbox_xyxyc, (1000, 1000), (1000, 1000))
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