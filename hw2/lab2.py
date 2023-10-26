#%%
'''
NOTE: see the `run.sh` script for the actual commands used to run the experiments. This script is just the code for the experiments.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tqdm 
import argparse
import time 
import sys 
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--run-test", action="store_true", default=False)
parser.add_argument("--disable-batchnorm", action="store_true", default=False)
parser.add_argument("--profile", action="store_true", default=False)
parser.add_argument("--profiler-name", type = str, default="resnet18")
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, disable_batchnorm=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if disable_batchnorm:
            self.bn1 = Identity()
            self.bn2 = Identity() 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if not disable_batchnorm else Identity()
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, disable_batchnorm=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        ## preset
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if disable_batchnorm:
            self.bn1 = Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, disable_batchnorm=disable_batchnorm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, disable_batchnorm=disable_batchnorm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, disable_batchnorm=disable_batchnorm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, disable_batchnorm=disable_batchnorm)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, disable_batchnorm=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, disable_batchnorm=disable_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(disable_batchnorm=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], disable_batchnorm=disable_batchnorm)


class Trainer:
    def __init__(self, args):
        self.cnn = ResNet18(args.disable_batchnorm)
        self.cnn.to(args.device)
        self.device = args.device
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.run_test = args.run_test
        self.dl_time_elapsed = 0
        self.dl_time_counter = 0
        self.mb_time_elapsed = 0
        self.mb_time_counter = 0
        self.epoch_time_elapsed = 0
        self.epoch_time_counter = 0
        self.last_total_accuracy=0
        self.profile = args.profile
        self.profiler_name = args.profiler_name



        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.cnn.parameters(), lr=args.lr,
                                             momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "nesterov":
            self.optimizer = torch.optim.SGD(self.cnn.parameters(), lr=args.lr,
                                             momentum=args.momentum, weight_decay=args.weight_decay,
                                             nesterov=True)
        elif args.optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.cnn.parameters(), lr=args.lr,
                                                 weight_decay=args.weight_decay)
        elif args.optimizer == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.cnn.parameters(), lr=args.lr,
                                                  weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(f"optimizer {args.optimizer} not implemented")
        
    def get_train_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='data/cifar10/train', train=True, download=True, transform=transform_train)
        train_dl = torch.utils.data.DataLoader(
            trainset,
            batch_size = 128,
            shuffle = True, 
            drop_last = True,
            num_workers = self.num_workers,
            pin_memory = self.device == "cuda"
            )
        return train_dl
    def get_test_data(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='data/cifar10/test', train=False, download=True, transform=transform_test)

        test_dl = torch.utils.data.DataLoader(
            shuffle=False,
            batch_size = 100,
            drop_last = False,
            dataset = testset,
            pin_memory = self.device == "cuda"
            )
        return test_dl
    def test_loop(self):
        self.cnn.eval()
        test_dl = self.get_test_data()
        for x,y in test_dl:
            with torch.no_grad():
                return self.cnn(x)
    def train_loop(self, train_dl):
        
        outstr = ""
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_logs/{self.profiler_name}'),
            record_shapes=True,
            with_stack=True)
        if self.profile:
            prof.start()
        ## skip first epoch

        for i in range(self.epochs):
            
            total_correct=0
            total_samples=0
            total_loss = 0
            self.cnn.train()
            n_batches = len(train_dl)
            train_dl_i = iter(train_dl)
            
            torch.cuda.synchronize()
            epoch_start = time.perf_counter_ns()
            for j in range(n_batches):
                if self.profile:
                    prof.step()
                
                torch.cuda.synchronize()
                dl_start = time.perf_counter_ns()
                x,y = next(train_dl_i)
                
                torch.cuda.synchronize()
                dl_end = time.perf_counter_ns()
                dl_diff = dl_end - dl_start
                self.dl_time_elapsed += dl_diff
                self.dl_time_counter += 1

                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                

                torch.cuda.synchronize()
                mb_start = time.perf_counter_ns()
                predictions = self.cnn(x)
                loss = F.cross_entropy(predictions,y)
                loss.backward()
                self.optimizer.step()
                       
                torch.cuda.synchronize()
                mb_end = time.perf_counter_ns()
                mb_diff = mb_end - mb_start
                self.mb_time_elapsed += mb_diff
                self.mb_time_counter += 1
                correct = (predictions.argmax(dim=1) == y).int().sum()
                accuracy = correct / len(y)
                total_correct += correct
                total_samples += len(y)
                total_loss += loss.item()

                print(f"\repoch {i}: loss {loss}; accuracy {accuracy}", end="", flush=True)
            
            torch.cuda.synchronize()
            epoch_end = time.perf_counter_ns()
            epoch_diff = epoch_end - epoch_start
            self.epoch_time_elapsed += epoch_diff
            self.epoch_time_counter += 1
            total_accuracy = total_correct / total_samples
            avg_loss = total_loss / n_batches
            outstr += f"epoch {i},{epoch_diff/1e9},{avg_loss},{total_accuracy}\n"
            if self.run_test:
                self.test_loop()
        if self.profile:
            prof.stop()
        return outstr
def main(args):
    trainer = Trainer(args)
    train_dl = trainer.get_train_data()
    outstr = trainer.train_loop(train_dl)
    avg_dl_time = trainer.dl_time_elapsed / trainer.dl_time_counter / 1e9
    avg_mb_time = trainer.mb_time_elapsed / trainer.mb_time_counter / 1e9
    avg_epoch_time = trainer.epoch_time_elapsed / trainer.epoch_time_counter /1e9
    print(f"\nAverage DataLoading time: {avg_dl_time} s")
    print(f"Average MiniBatch time: {avg_mb_time} s")
    print(f"Average epoch time: {avg_epoch_time} s")
    
    print(outstr)
    

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

        
