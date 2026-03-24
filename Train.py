import torch
import socket
import time
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from model.TLVNet import tlvnet
from torch.autograd import Variable
from model.data import get_training_set


parser = argparse.ArgumentParser(description='PyTorch TLV-Net')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--data_dir', type=str, default='dataset/train')
parser.add_argument('--label_train_dataset', type=str, default='reference')
parser.add_argument('--data_train_dataset', type=str, default='raw')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--T_max', type=int, default=50, help='CosineAnnealingLR cycle length')
parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--resume_train', type=bool, default=False)
parser.add_argument('--model', default='./weights/epoch_250.pth', help='Pretrained base model')


def train(epoch):
    torch.cuda.empty_cache()
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.to(device)
            target = target.to(device)

        t0 = time.time()
        model.forward(input, target, training=True)
        loss = model.elbo(target)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        epoch_loss += loss.item()
        optimizer.step()
        t1 = time.time()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || LR: {:.2e} || Timer: {:.4f} sec.".format(
            epoch, iteration, len(training_data_loader), loss.item(),
            optimizer.param_groups[0]['lr'], (t1 - t0)
        ))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def checkpoint(epoch):
    model_out_path = f"{opt.save_folder}/epoch_{epoch}.pth" if isinstance(epoch, int) \
                    else f"{opt.save_folder}/epoch_last.pth"
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


if __name__ == '__main__':
    opt = parser.parse_args()
    device = torch.device(opt.device)
    hostname = str(socket.gethostname())
    cudnn.benchmark = True

    print(opt)
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')

    train_set = get_training_set(
        opt.data_dir,
        opt.label_train_dataset,
        opt.data_train_dataset,
        opt.patch_size,
        opt.data_augmentation
    )
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True
    )

    model = tlvnet(opt)
    if opt.resume_train:
        model.load_state_dict(torch.load(opt.model, map_location=device))
        print(f'Loaded pretrained model from {opt.model}')
    if opt.gpu_mode:
        model = model.to(device)

    params = [
        {'params': model.decoder.parameters(), 'lr': opt.lr},
        {'params': model.DB.parameters(), 'lr': opt.lr * 0.5}
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-4)

    scheduler = lrs.CosineAnnealingLR(
        optimizer,
        T_max=opt.T_max,
        eta_min=1e-6
    )

    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)
        scheduler.step()

        if epoch % opt.snapshots == 0:
            checkpoint(epoch)
        if epoch == opt.nEpochs:
            checkpoint("last")
