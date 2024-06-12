import yaml
import os
import tqdm
import numpy
import torch
from utils import utils
from utils.dataset import Dataset
from torch.utils import data

config = {
    "local_rank": 0,
    "input_size": 640,
    "world_size": 1,
    'batch_size': 1,
    "epochs": 4
}
@torch.no_grad()
def demo(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import cv2
    palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=numpy.uint8)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    model = torch.load('./weights/best.pt', map_location=device)['model'].float()
    model.to(device)
    stride = int(max(model.stride.cpu().numpy()))

    model.half()
    model.eval()

    camera = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while camera.isOpened():
        # Capture frame-by-frame
        success, frame = camera.read()
        if success:
            image = frame.copy()
            shape = image.shape[:2]  # current shape [height, width]
            r = min(1.0, config['input_size'] / shape[0], config['input_size'] / shape[1])
            pad = int(round(shape[1] * r)), int(round(shape[0] * r))
            w = config['input_size'] - pad[0]
            h = config['input_size'] - pad[1]
            w = numpy.mod(w, stride)
            h = numpy.mod(h, stride)
            w /= 2
            h /= 2
            if shape[::-1] != pad:  # resize
                image = cv2.resize(image,
                                   dsize=pad,
                                   interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image,
                                       top, bottom,
                                       left, right,
                                       cv2.BORDER_CONSTANT)  # add border
            # Convert HWC to CHW, BGR to RGB
            image = image.transpose((2, 0, 1))[::-1]
            image = numpy.ascontiguousarray(image)
            image = torch.from_numpy(image)
            image = image.unsqueeze(dim=0)
            image = image.to(device)
            image = image.half()
            image = image / 255
            # Inference
            outputs = model(image)
            # NMS
            outputs = utils.non_max_suppression(outputs, 0.25, 0.7, model.head.nc)
            for output in outputs:
                output = output.clone()
                if len(output):
                    box_output = output[:, :6]
                    kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
                else:
                    box_output = output[:, :6]
                    kps_output = output[:, 6:]

                r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

                box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                box_output[:, :4] /= r

                box_output[:, 0].clamp_(0, shape[1])  # x
                box_output[:, 1].clamp_(0, shape[0])  # y
                box_output[:, 2].clamp_(0, shape[1])  # x
                box_output[:, 3].clamp_(0, shape[0])  # y

                kps_output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                kps_output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                kps_output[..., 0] /= r
                kps_output[..., 1] /= r
                kps_output[..., 0].clamp_(0, shape[1])  # x
                kps_output[..., 1].clamp_(0, shape[0])  # y

                for box in box_output:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    cv2.rectangle(frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                for kpt in reversed(kps_output):
                    for i, k in enumerate(kpt):
                        color_k = [int(x) for x in kpt_color[i]]
                        x_coord, y_coord = k[0], k[1]
                        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                            if len(k) == 3:
                                conf = k[2]
                                if conf < 0.5:
                                    continue
                            cv2.circle(frame,
                                       (int(x_coord), int(y_coord)),
                                       5, color_k, -1, lineType=cv2.LINE_AA)
                    for i, sk in enumerate(skeleton):
                        pos1 = (int(kpt[(sk[0] - 1), 0]), int(kpt[(sk[0] - 1), 1]))
                        pos2 = (int(kpt[(sk[1] - 1), 0]), int(kpt[(sk[1] - 1), 1]))
                        if kpt.shape[-1] == 3:
                            conf1 = kpt[(sk[0] - 1), 2]
                            conf2 = kpt[(sk[1] - 1), 2]
                            if conf1 < 0.5 or conf2 < 0.5:
                                continue
                        if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                            continue
                        if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                            continue
                        cv2.line(frame,
                                 pos1, pos2,
                                 [int(x) for x in limb_color[i]],
                                 thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    camera.release()

    # Closes all the frames
    cv2.destroyAllWindows()

demo(config)