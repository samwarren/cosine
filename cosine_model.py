import argparse
import logging
import math
import sys

from ignite.metrics import Loss
from tensorboardX import SummaryWriter
import torch
from torch.nn import ModuleList
from torch.utils import data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_data(examples=180, noise_ratio=0.1):
    x = torch.unsqueeze(torch.linspace(-1 * math.pi, math.pi, examples), dim=1)
    y = torch.cos(x) + noise_ratio * torch.rand(x.size())
    return data.TensorDataset(x, y)


def create_loader(examples=128, noise_ratio=0.1, batch_size=64, shuffle=True):
    dataset = create_data(examples=examples, noise_ratio=noise_ratio)
    loader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=2, shuffle=shuffle
    )
    return loader


class CosineNet(torch.nn.Module):
    def __init__(self, writer, num_hidden_layers=1):
        super(CosineNet, self).__init__()
        input_features = 1
        hidden_output_features = 10
        final_output_features = 1
        self.writer = writer
        layers = ModuleList()
        for i in range(num_hidden_layers):
            if i == 0:
                layers.append(torch.nn.Linear(input_features, hidden_output_features))
            else:
                layers.append(
                    torch.nn.Linear(hidden_output_features, hidden_output_features)
                )
            layers.append(torch.nn.ReLU())
        final_layer = torch.nn.Linear(hidden_output_features, final_output_features)
        layers.append(final_layer)
        self.model = torch.nn.Sequential(*layers)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, input):
        preds = self.model(input)
        return preds

    def evaluate(self, data_loader, epoch=0):
        loss_metric = Loss(self.loss_func)
        for i, (inputs, targets) in enumerate(data_loader):
            preds = self.model.forward(inputs)
            loss = self.loss_func(preds, targets)
            loss_metric.update((preds, targets))
            logger.debug(f"validation: {i} has loss {loss.item()}")
        mean_epoch_loss = loss_metric.compute()
        logger.info(f"validation mean loss is {mean_epoch_loss}")
        self.writer.add_scalar("val/loss", mean_epoch_loss, epoch)

    def train_loop(self, train_loader, val_loader, epochs=1000, lr=0.01):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for i in range(epochs):
            self.train_one_epoch(
                dataloader=train_loader,
                epoch=i,
                model=self.model,
                loss_func=self.loss_func,
                optimizer=optimizer,
            )
            self.evaluate(data_loader=val_loader, epoch=i)

    def train_one_epoch(self, dataloader, model, epoch, loss_func, optimizer):
        loss_metric = Loss(self.loss_func)
        for i, (inputs, targets) in enumerate(dataloader):
            preds = model.forward(inputs)
            loss = loss_func(preds, targets)
            loss_metric.update((preds, targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.debug(f"epoch: {epoch}, batch: {i} has training loss {loss.item()}")
        mean_epoch_loss = loss_metric.compute()
        logger.info(f"epoch {epoch}, has mean training loss is {mean_epoch_loss}")
        self.writer.add_scalar("training/loss", mean_epoch_loss, epoch)


def parse_args():
    command_parser = argparse.ArgumentParser(description="CosineNet Modeling Interface")
    command_parser.add_argument(
        "command", help="sub-command to run", choices=("train", "evaluate")
    )
    parser = argparse.ArgumentParser(description="CosineNet Modeling Interface")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="turn on verbose mode to enable debug logging",
    )
    parser.add_argument(
        "--logdir", type=str, default=None, help="path where to save training logs"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for ADAM")
    parser.add_argument(
        "--training-noise",
        type=float,
        default=0.1,
        help="amount of noise to add to training data",
    )
    parser.add_argument(
        "--val-noise",
        type=float,
        default=0.1,
        help="amount of noise to add to val data",
    )
    parser.add_argument(
        "--n-hidden-layers",
        type=int,
        default=3,
        help="number of hidden layers for model",
    )
    parser.add_argument(
        "--n-val-examples",
        type=int,
        default=2048,
        help="number of examples in validation set",
    )
    parser.add_argument(
        "--n-train-examples",
        type=int,
        default=256,
        help="number of examples in training set",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--n-training-epochs",
        type=int,
        default=1000,
        help="number of epochs to train for",
    )
    command = command_parser.parse_args(sys.argv[1:2]).command
    args = parser.parse_args(sys.argv[2:])
    logger.info("Parsed CLI args: %s", args)
    writer = SummaryWriter(args.logdir)
    cosine_net = CosineNet(writer=writer, num_hidden_layers=args.n_hidden_layers)
    val_loader = create_loader(
        examples=args.n_val_examples,
        noise_ratio=args.val_noise,
        batch_size=args.batch_size,
        shuffle=False,
    )
    if command == "train":
        train_loader = create_loader(
            examples=args.n_train_examples,
            noise_ratio=args.training_noise,
            batch_size=args.batch_size,
            shuffle=True,
        )

        cosine_net.train_loop(
            train_loader, val_loader, epochs=args.n_training_epochs, lr=args.lr
        )
    elif command == "evaluate":
        cosine_net.evaluate(val_loader)

    writer.close()


if __name__ == "__main__":
    parse_args()

# todo tune hyperparams: num hidden layers, learning rate
