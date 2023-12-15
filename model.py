import os
import pickle
from time import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from pre_process import PlanNode, Preprocessor
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)
from TreeConvolution.util import prepare_trees

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cpu")


def collate_pairwise_fn(batch):
    trees1, trees2, labels = zip(*batch)
    return list(trees1), list(trees2), list(labels)


def transformer(x: PlanNode):
    return x.get_feature()


def left_child(x: PlanNode):
    return x.get_left_child()


def right_child(x: PlanNode):
    return x.get_left_child()


def build_trees(feature):
    return prepare_trees(feature, transformer, left_child, right_child, device=device)


def generate_network(input_feature_dim):
    return nn.Sequential(
        BinaryTreeConv(input_feature_dim, 256),
        TreeLayerNorm(),
        TreeActivation(nn.LeakyReLU()),
        BinaryTreeConv(256, 128),
        TreeLayerNorm(),
        TreeActivation(nn.LeakyReLU()),
        BinaryTreeConv(128, 64),
        TreeLayerNorm(),
        DynamicPooling(),
        nn.Linear(64, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 1)
    )


class LeroModelPairWise(nn.Module):
    def __init__(self, feature_generator: Union[Preprocessor, None],
                 model_path: str = None, batch_size: int = 64, num_epochs: int = 1):
        super().__init__()
        self.net = None
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_generator = feature_generator

        if model_path:
            self.load(model_path)

    def fit(self, x1, x2, y1, y2):
        y1 = np.array(y1).reshape(-1, 1)
        y2 = np.array(y2).reshape(-1, 1)

        self.feature_generator.input_feature_dim = len(x1[0].get_feature())
        print("input_feature_dim:", self.feature_generator.input_feature_dim)

        self.net = generate_network(self.feature_generator.input_feature_dim)

        pairs = []
        for i in range(len(x1)):
            pairs.append((x1[i], x2[i], 1.0 if y1[i] >= y2[i] else 0.0))

        dataset = DataLoader(pairs, batch_size=self.batch_size, shuffle=True, collate_fn=collate_pairwise_fn)
        optimizer = torch.optim.Adam(self.net.parameters())

        bce_loss_fn = torch.nn.BCELoss()

        sigmoid = nn.Sigmoid()
        start_time = time()
        for epoch in range(self.num_epochs):
            losses, itr = 0, 0
            for x1, x2, label in dataset:
                prob_y = sigmoid(self.net(build_trees(x1)) - self.net(build_trees(x2)))
                label_y = torch.tensor(np.array(label).reshape(-1, 1))

                loss = bce_loss_fn(prob_y, label_y)
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if itr % 100 == 0:
                    print("Epoch", epoch, "Iteration", itr, "loss:", loss.item())

                itr += 1

            losses /= len(dataset)

            print("Epoch", epoch, "training loss:", losses)
        print("training time:", time() - start_time, "batch size:", self.batch_size)

    def forward(self, x):
        if self.net is None:
            raise Exception("Either train the model or load from a checkpoint to get predictions!!")

        return self.net(build_trees(x)).cpu().detach().numpy()

    def load(self, path):
        with open(os.path.join(path, "feature_generator.pickle"), "rb") as f:
            self.feature_generator = pickle.load(f)

        self.net = generate_network(self.feature_generator.input_feature_dim)
        self.net.load_state_dict(torch.load(os.path.join(path, "nn_weights"), map_location=torch.device('cpu')))
        self.net.eval()

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self.net.state_dict(), os.path.join(path, "nn_weights"))

        with open(os.path.join(path, "feature_generator.pickle"), "wb") as f:
            pickle.dump(self.feature_generator, f)
