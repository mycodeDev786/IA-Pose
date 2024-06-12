import yaml
import os
import csv
import tqdm
import copy
import numpy
from nets import nn
import torch
from utils import utils
from utils.dataset import Dataset
from torch.utils import data

config = {
    "local_rank": 0,
    "input-size": 640,
    "world_size": 1,
    'batch_size': 1,
    "epochs": 1
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
            correct_box = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device) # init
            correct_kpt = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device) # init
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


class Train:
    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    model = nn.yolo_v8_n(len(params['names']))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)  # Or model.cuda() for default device
    else:
        device = torch.device("cpu")
        model.to(device)
    # load optimizer
    accumulate = max(round(64 / (config['batch_size'] * config['world_size'])), 1)
    params['weight_decay'] *= config['batch_size'] * config['world_size'] * accumulate / 64
    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)
    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p
    # Scheduler
    lr = learning_rate(config['epochs'], params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)
    # EMA
    ema = utils.EMA(model) if config['local_rank'] == 0 else None
    image_dir = "./dataset/images/train/"
    filenames = []
    for filename in os.listdir(image_dir):
        # Append the full path of each file to the list
        full_path = os.path.join(image_dir, filename)
        filenames.append(full_path)

    dataset = Dataset(filenames, config['input-size'], params, True)
    if config['world_size'] <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)
    loader = data.DataLoader(dataset, config['batch_size'], sampler is None, sampler,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    if config['world_size'] > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[config['local_rank']],
                                                          output_device=config['local_rank'])
    best = 0
    num_batch = len(loader)
    if torch.cuda.is_available():
        amp_scale = torch.cuda.amp.GradScaler()
    else:
        amp_scale = None
    criterion = utils.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    with open('weights/step.csv', 'w') as f:
        if config['local_rank'] == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'BoxAP', 'PoseAP'])
            writer.writeheader()
        for epoch in range(config['epochs']):
            model.train()
            if config['epochs'] - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = utils.AverageMeter()
            if config['world_size'] > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if config['local_rank'] == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
            if config['local_rank'] == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()
            for i, (samples, targets) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.to(device).float() / 255

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (config['batch_size'] * config['world_size'])]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)
                # Forward
                if torch.cuda.is_available():
                    from torch.cuda.amp import autocast
                    device = torch.device("cuda")
                    with autocast():
                        outputs = model(samples)  # forward
                        loss = criterion(outputs, targets)
                else:
                    device = torch.device("cpu")
                    outputs = model(samples)  # forward
                    loss = criterion(outputs, targets)
                m_loss.update(loss.item(), samples.size(0))
                loss *= config['batch_size']  # loss scaled by batch_size
                loss *= config['world_size']  # gradient averaged between devices in DDP mode
                # Backward
                if amp_scale is not None:
                    amp_scale.scale(loss).backward()
                    amp_scale.step(optimizer)
                    amp_scale.update()
                else:
                    loss.backward()
                if x % accumulate == 0:
                    if amp_scale is not None:
                        amp_scale.unscale_(optimizer)  # unscale gradients

                    utils.clip_gradients(model)  # clip gradients

                    if amp_scale is not None:
                        amp_scale.step(optimizer)  # optimizer.step
                        amp_scale.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                    if ema:
                        ema.update(model)
                        # Log
                if config['local_rank'] == 0:
                    s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{config['epochs']}', m_loss.avg)
                    p_bar.set_description(s)
                del loss
                del outputs
            scheduler.step()
            if config['local_rank'] == 0:
                last = test(params, ema.ema)
                writer.writerow({'epoch': str(epoch + 1).zfill(3),
                         'BoxAP': str(f'{last[0]:.3f}'),
                         'PoseAP': str(f'{last[1]:.3f}')})
            f.flush()

        # Update best mAP
            if last[1] > best:
              best = last[1]

        # Save model
            ckpt = {'model': copy.deepcopy(ema.ema).half()}

        # Save last, best and delete
            torch.save(ckpt, './weights/last.pt')
            if best == last[1]:
              torch.save(ckpt, './weights/best.pt')
            del ckpt

    if config['local_rank'] == 0:
        utils.strip_optimizer('./weights/best.pt')  # strip optimizers
        utils.strip_optimizer('./weights/last.pt')  # strip optimizers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

