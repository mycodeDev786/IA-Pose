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
    "input-size": 640,
    "world_size": 1,
    'batch_size': 1,
    "epochs": 4
}


def learning_rate(epochs, params):
    def fn(x):
        return (1 - x / epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


@torch.no_grad()
def test(params, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = "./dataset/images/test/"
    filenames = []
    for filename in os.listdir(image_dir):
        # Append the full path of each file to the list
        full_path = os.path.join(image_dir, filename)
        filenames.append(full_path)

    numpy.random.shuffle(filenames)
    dataset = Dataset(filenames, 640, params, False)
    loader = data.DataLoader(dataset, 1, False, num_workers=0,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    if model is None:
        model = torch.load('./weights/best.pt', map_location=device)['model'].float()
    model.half()
    model.eval()

    iou_v = torch.linspace(0.5, 0.95, 10).to(device)
    n_iou = iou_v.numel()

    box_mean_ap = 0.
    kpt_mean_ap = 0.
    box_metrics = []
    kpt_metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 2) % ('BoxAP', 'PoseAP'))
    for samples, targets in p_bar:
        samples = samples.to(device)
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch size, channels, height, width
        scale = torch.tensor((w, h, w, h)).to(device)
        # Inference
        outputs = model(samples)
        # NMS
        outputs = utils.non_max_suppression(outputs, 0.001, 0.7, model.head.nc)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]
            kpt = targets['kpt'][idx]

            cls = cls.to(device)
            box = box.to(device)
            kpt = kpt.to(device)

            correct_box = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)  # init
            correct_kpt = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)  # init
            if output.shape[0] == 0:
                if cls.shape[0]:
                    box_metrics.append((correct_box,
                                        *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                    kpt_metrics.append((correct_kpt,
                                        *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue

            # Predictions
            pred = output.clone()
            p_kpt = pred[:, 6:].view(output.shape[0], kpt.shape[1], -1)

            # Evaluate
            if cls.shape[0]:
                t_box = utils.wh2xy(box)
                t_kpt = kpt.clone()
                t_kpt[..., 0] *= w
                t_kpt[..., 1] *= h

                target = torch.cat((cls, t_box * scale), 1)  # native-space labels
                correct_box = utils.compute_metric(pred[:, :6], target, iou_v)
                correct_kpt = utils.compute_metric(pred[:, :6], target, iou_v, p_kpt, t_kpt)
            # Append
            box_metrics.append((correct_box, output[:, 4], output[:, 5], cls.squeeze(-1)))
            kpt_metrics.append((correct_kpt, output[:, 4], output[:, 5], cls.squeeze(-1)))

    box_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*box_metrics)]  # to numpy
    kpt_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*kpt_metrics)]  # to numpy
    if len(box_metrics) and box_metrics[0].any():
        tp, fp, m_pre, m_rec, map50, box_mean_ap = utils.compute_ap(*box_metrics)
    if len(kpt_metrics) and kpt_metrics[0].any():
        tp, fp, m_pre, m_rec, map50, kpt_mean_ap = utils.compute_ap(*kpt_metrics)
    # Print results

    print('%10.3g' * 2 % (box_mean_ap, kpt_mean_ap))

    # Return results
    model.float()  # for training

    return box_mean_ap, kpt_mean_ap


with open('utils/args.yaml', errors='ignore') as f:
    params = yaml.safe_load(f)
test(params)
