import association, kalmanfilter, ocsort


bboxes1 = [
    [100, 200, 300, 400],
    [150, 230, 200, 450]
]

bboxes2 = [
    [50, 60, 350, 200],
    [10, 20, 100, 800],
    [200, 400, 500, 900],
    [80, 800, 90, 850],
    [650, 550, 750, 600]
]

iou_batch = association.iou_batch(bboxes1, bboxes2)
print(iou_batch)