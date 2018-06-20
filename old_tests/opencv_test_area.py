import math

import cv2
import tqdm
import numpy as np

PADCOLOR = [255, 255, 255]
drone_im = cv2.resize(cv2.imread("drone.png"), (0, 0), fx=0.08, fy=0.08)
num_frames = 500
preds = np.zeros((num_frames, 3))
labels = np.zeros((num_frames, 3))
dist = 0.8
head = -45
delta = 1
for i in range(0, preds.shape[0]):
    labels[i, 0] = dist
    labels[i, 1] = head
    labels[i, 2] = delta
    dist = dist + 0.0005
    head = head + 0.2
    delta = delta - 0.01

dist = 1.437
head = -0
delta = 0.9
for i in range(0, preds.shape[0]):
    preds[i, 0] = dist
    preds[i, 1] = head
    preds[i, 2] = delta
    dist = dist + 0.0005
    head = head + 0.2
    delta = delta - 0.01


def frame_composer(i):
    # Adjusting the image
    img_f = np.random.randint(0, 256, (60, 107, 3), dtype=np.uint8)
    scaled = cv2.resize(img_f, (0, 0), fx=4, fy=4)
    vert_p = int((480 - scaled.shape[0]) / 2)

    hor_p = int((640 - scaled.shape[1]) / 2)
    im_pad = cv2.copyMakeBorder(scaled,
                                vert_p,
                                vert_p if vert_p * 2 + scaled.shape[0] == 480 else vert_p + (480 - (vert_p * 2 + scaled.shape[0])),
                                hor_p,
                                hor_p if hor_p * 2 + scaled.shape[1] == 640 else hor_p + (640 - (hor_p * 2 + scaled.shape[1])),
                                cv2.BORDER_CONSTANT, value=PADCOLOR)
    im_partial = cv2.cvtColor(im_pad, cv2.COLOR_RGB2BGR)
    data_area = (np.ones((480, 640, 3)) * 255).astype(np.uint8)
    im_final = np.hstack((data_area, im_partial))

    # Setting some variables
    font = cv2.FONT_HERSHEY_DUPLEX
    text_color = (0, 0, 0)
    y_d = preds[i]
    l_d = labels[i]
    cv2.putText(im_final, "Frame: %s" % i, (900, 50), font, 0.5, text_color, 1, cv2.LINE_AA)

    # Top view
    triangle_color = (255, 229, 204)
    angle_deg = y_d[1]

    # Text Information
    cv2.putText(im_final, "Distance T: %.3f" % (l_d[0]), (20, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "Distance P: %.3f" % (y_d[0]), (20, 40), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "Angle T: %.3f" % (l_d[1]), (220, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "Angle P: %.3f" % angle_deg, (220, 40), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "Delta z T: %.3f" % (l_d[2]), (400, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "Delta z P: %.3f" % (y_d[2]), (400, 40), font, 0.5, text_color, 1, cv2.LINE_AA)

    # draw legend
    pr_color = (255, 0, 0)

    gt_color = (0, 255, 0)

    cv2.putText(im_final, "Truth", (420, 425), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.circle(im_final, center=(405, 420), radius=5, color=gt_color, thickness=2)

    cv2.putText(im_final, "Prediction", (420, 455), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.circle(im_final, center=(405, 450), radius=5, color=pr_color, thickness=5)

    # Draw FOV and drone
    t_x = 330
    t_y = 400
    camera_fov = 90
    triangle_side_len = 400
    x_offset = t_x - drone_im.shape[0] / 2 - 4
    y_offset = t_y - 7
    im_final[y_offset:y_offset + drone_im.shape[0], x_offset:x_offset + drone_im.shape[1]] = drone_im
    triangle = np.array([[int(t_x - (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), int(t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len))],
                         [t_x, t_y],
                         [int(t_x + (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), int(t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len))]], np.int32)
    cv2.fillConvexPoly(im_final, triangle, color=triangle_color, lineType=1)
    scale_factor = (math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / (2 * 1.437)

    cv2.putText(im_final, "Relative pose", (300, 90), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(im_final,
             (30, int(t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len))),
             (30, t_y),
             color=(0, 0, 0),
             thickness=1)

    cv2.line(im_final,
             (15, int((t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len)))),
             (30, int((t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len)))),
             color=(0, 0, 0),
             thickness=1)
    cv2.putText(im_final, "2.8 m", (31, int((t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len)))), font, 0.4, text_color, 1, cv2.LINE_AA)
    cv2.line(im_final,
             (15, int((t_y - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))),
             (30, int((t_y - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))),
             color=(0, 0, 0),
             thickness=1)
    cv2.putText(im_final, "1.4 m", (31, int((t_y+3 - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))), font, 0.4, text_color, 1, cv2.LINE_AA)
    cv2.line(im_final,
             (15, t_y),
             (30, t_y),
             color=(0, 0, 0),
             thickness=1)
    cv2.putText(im_final, "0 m", (31, t_y + 5), font, 0.4, text_color, 1, cv2.LINE_AA)

    # draw GT
    gt_center = (int((t_x + scale_factor * (math.sin(math.radians(labels[i, 1])) * labels[i, 0]))),
                 int((t_y - scale_factor * (math.cos(math.radians(labels[i, 1])) * labels[i, 0]))))
    cv2.circle(im_final, center=gt_center, radius=5, color=gt_color, thickness=2)

    # draw Pred
    pr_center = (int((t_x + scale_factor * (math.sin(math.radians(preds[i, 1])) * preds[i, 0]))),
                 int((t_y - scale_factor * (math.cos(math.radians(preds[i, 1])) * preds[i, 0]))))
    cv2.circle(im_final, center=pr_center, radius=5, color=pr_color, thickness=5)

    # draw height

    h_x = 640
    h_y = 90
    cv2.putText(im_final, "Delta Height", (h_x, h_y), font, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "+1", (h_x + 65, h_y + 15), font, 0.4, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "0", (h_x + 65, h_y + 164), font, 0.4, text_color, 1, cv2.LINE_AA)
    cv2.putText(im_final, "-1", (h_x + 65, h_y + 312), font, 0.4, text_color, 1, cv2.LINE_AA)

    cv2.rectangle(im_final, (h_x + 30, h_y + 10), (h_x + 60, h_y + 310), color=(0, 0, 0), thickness=2)
    cv2.line(im_final, (h_x + 35, h_y + 160), (h_x + 55, h_y + 160), color=(0, 0, 0), thickness=1)
    h_c_x = h_x+45
    h_c_y = h_y + 160

    h_scale_factor = 300 / 2

    gt_h_center = (h_c_x,
                   int((h_c_y - h_scale_factor * labels[i, 2])))
    pr_h_center = (h_c_x,
                   int((h_c_y - h_scale_factor * preds[i, 2])))
    cv2.circle(im_final, center=gt_h_center, radius=5, color=gt_color, thickness=2)
    cv2.circle(im_final, center=pr_h_center, radius=5, color=pr_color, thickness=5)

    video_writer.write(im_final)


fps = 30
width = 1280
height = 480
video_writer = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

max_ = preds.shape[0]
for i in tqdm.tqdm(range(0, max_)):
    frame_composer(i)
video_writer.release()
cv2.destroyAllWindows()


import numpy as np
import cv2

asd =np.ones((480, 856, 3),dtype=np.uint8)*145
video_writer=cv2.VideoWriter("asd.avi",cv2.VideoWriter_fourcc(*'XVID'),30,(856,480 ))
for i in range(300):
    video_writer.write(asd)
video_writer.release()
cv2.destroyAllWindows()