import argparse

import torch
from torch.utils.data import DataLoader

from dataset.predict import VideoPredict
from model.network import C3DVQANet


def predict_model(model, device, dataloaders):

    phase = 'predict'
    model.eval()

    epoch_preds = []

    for ref, dis, in dataloaders[phase]:
        ref = ref.to(device)
        dis = dis.to(device)

        # dim: [batch=1, P, C, D, H, W]
        ref = ref.reshape(-1, ref.shape[2], ref.shape[3], ref.shape[4], ref.shape[5])
        dis = dis.reshape(-1, dis.shape[2], dis.shape[3], dis.shape[4], dis.shape[5])

        with torch.no_grad():
            preds = model(ref, dis)
            preds = torch.mean(preds, 0, keepdim=True)
            print(preds)
            epoch_preds.append(preds.flatten())

    epoch_preds = torch.cat(epoch_preds).flatten().data.cpu().numpy()

    epoch_preds = epoch_preds.tolist()
    print(sum(epoch_preds) / len(epoch_preds))


def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_ref', type=str, help='Path to reference video')
    parser.add_argument('--video_dis', type=str, help='Path to distorion video')
    parser.add_argument('--load_model', type=str, required=True, help='Path to load checkpoint')
    parser.add_argument('--log_file_name', default='./log/run.log', type=str, help='Path to save log')

    parser.add_argument('--scene_nb_frames', default=100, type=int, help='number of frames for each scene')
    parser.add_argument('--channel',
                        default=1,
                        type=int,
                        help='channel number of input data, 1 for Y channel, 3 for YUV')
    parser.add_argument('--size_x', default=112, type=int, help='patch size x of segment')
    parser.add_argument('--size_y', default=112, type=int, help='patch size y of segment')
    parser.add_argument('--stride_x', default=80, type=int, help='patch stride x between segments')
    parser.add_argument('--stride_y', default=80, type=int, help='patch stride y between segments')

    parser.add_argument('--multi_gpu', action='store_true', help='whether to use all GPUs')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    opt = parse_opts()

    video_ref = opt.video_ref
    video_dis = opt.video_dis
    load_checkpoint = opt.load_model
    MULTI_GPU_MODE = opt.multi_gpu
    channel = opt.channel
    size_x = opt.size_x
    size_y = opt.size_y
    stride_x = opt.stride_x
    stride_y = opt.stride_y

    video_dataset = {
        x:
        VideoPredict(video_ref,
                     video_dis,
                     channel=channel,
                     size_x=size_x,
                     size_y=size_y,
                     stride_x=stride_x,
                     stride_y=stride_y)
        for x in ['predict']
    }
    dataloaders = {
        x: DataLoader(video_dataset[x], batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        for x in ['predict']
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(load_checkpoint)

    model = C3DVQANet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))
    else:
        print('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))

    predict_model(model, device, dataloaders)
