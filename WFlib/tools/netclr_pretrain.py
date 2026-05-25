import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.cuda.amp import GradScaler, autocast
import tqdm


def cal_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class NetCLR(object):
    def __init__(self, **args):
        self.model = args["model"]
        self.optimizer = args["optimizer"]
        self.scheduler = args["scheduler"]
        self.fp16_precision = args["fp16_precision"]
        self.num_epoches = args["num_epoches"]
        self.batch_size = args["batch_size"]
        self.temperature = args["temperature"]
        self.out_file = args["out_file"]
        self.device = args.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_views = 2
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.log_every_n_step = 100

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size, device=self.device) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.fp16_precision)
        n_iter = 0
        print(f"Start SimCLR training for {self.num_epoches} number of epoches")

        for epoch_counter in range(self.num_epoches + 1):
            with tqdm.tqdm(train_loader, unit="batch") as tepoch:
                for data, _ in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")

                    self.model.train()

                    data = torch.cat(data, dim=0)
                    data = data.view(data.size(0), 1, data.size(1))
                    data = data.float().to(self.device, non_blocking=True)

                    with autocast(enabled=self.fp16_precision):
                        features = self.model(data)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if n_iter % self.log_every_n_step == 0:
                        top1, top5 = cal_accuracy(logits, labels, topk=(1, 5))
                        tepoch.set_postfix(loss=loss.item(), accuracy=top1.item())

                    n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()

            if epoch_counter % 50 == 0:
                torch.save(self.model.state_dict(), self.out_file)


class PreTrainData(Dataset):
    def __init__(self, x_train, y_train, augmentor, n_views):
        self.x = x_train
        self.y = y_train
        self.augmentor = augmentor
        self.n_views = n_views

    def _aug(self, inp):
        flip_idx = np.random.randint(0, 4999, 250)
        x_w = inp.copy()
        temp = x_w[flip_idx]
        x_w[flip_idx] = x_w[flip_idx + 1]
        x_w[flip_idx + 1] = temp
        return x_w

    def __getitem__(self, index):
        return [self.augmentor.augment(self.x[index]) for _ in range(self.n_views)], self.y[index]

    def __len__(self):
        return len(self.x)