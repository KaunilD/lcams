from SegNet import SegNet
from VGGNet import VGGNet
from Alex import Alex
from mlp import Model
from classifier import Classifier
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from evaluator import Evaluator
from chainer.datasets import tuple_dataset
import copy
lr = 0.01
epochs = 1000
bs = 100

def main():
    model = VGGNet()
    model.train = True

    model = Classifier(model)
    model0 = copy.deepcopy(model)
    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)
    model3 = copy.deepcopy(model)
    model0.to_gpu(0)
    model1.to_gpu(1)
    model2.to_gpu(2)
    model3.to_gpu(3)
    ds = np.load("/home/kaunildhruv/fbsource/fbcode/experimental/themachinist/ml/autoencoders/preprocess_ds.npz")
    print("Dataset loaded.")
    train, test = tuple_dataset.TupleDataset(ds["train_img"], ds["train_lable"]), tuple_dataset.TupleDataset(ds["test_img"], ds["test_lable"])

    train_iter = iterators.SerialIterator(train, batch_size=bs, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=bs, shuffle=False, repeat=False)
    optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)

    updater = training.ParallelUpdater(train_iter, optimizer, devices={'main': 0, 'first': 1, 'second': 2, 'third':3})
    trainer = training.Trainer(updater, (epochs, 'epoch'), out='result')
    #trainer.extend(Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    interval = (5, 'epoch')
    iter_interval = (10000, 'iteration')

    trainer.extend(extensions.snapshot_object(model, 'epoch-{.updater.epoch}.model'), trigger=interval)
    trainer.extend(extensions.snapshot_object(model, 'iteration-{.updater.iteration}.model'), trigger=iter_interval)
    trainer.extend(extensions.snapshot(), trigger=interval)

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
if __name__=="__main__":
    main()
