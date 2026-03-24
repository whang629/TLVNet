import os
import torch
import cv2
import time
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.TLVNet import tlvnet
from model.data import get_eval_set
from multiprocessing import freeze_support


parser = argparse.ArgumentParser(description='PyTorch TLV-Net')
parser.add_argument('--testBatchSize', type=int, default=1, help='Test batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='Number of data loading threads')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--input_dir', type=str, default='./dataset/UIEB-140/test-140', help='Input directory')
parser.add_argument('--output', default='./results', help='Result save path')
parser.add_argument('--model', default='weights/epoch_250.pth', help='Path to the pre - trained model')

opt = parser.parse_args()
print(opt)
device = torch.device(opt.device)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("GPU not found. Please run without using the --cuda parameter.")


print('===> Load the dataset')
test_set = get_eval_set(opt.input_dir, opt.input_dir)

testing_data_loader = DataLoader(
    dataset=test_set,
    num_workers=0,
    batch_size=opt.testBatchSize,
    shuffle=False
)


print('===> Build the model')
model = tlvnet(opt)


if cuda:
    model.load_state_dict(torch.load(opt.model, map_location=device))
else:
    model.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
print('The pre - trained model has been loaded.')

if cuda:
    model = model.to(device)


def eval():
    model.eval()
    torch.set_grad_enabled(False)
    for batch in testing_data_loader:
        with torch.no_grad():
            input, _, name = Variable(batch[0]), Variable(batch[1]), batch[2]
            if cuda:
                input = input.to(device)

            t0 = time.time()
            prediction = model.sample(input, testing=True)
            t1 = time.time()

            save_img = prediction.squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            save_dir = opt.output
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.basename(name[0])
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB))

            print(f"Processing completed: {filename} || Elapsed time: {t1 - t0:.4f} 秒")


if __name__ == '__main__':
    freeze_support()
    eval()
