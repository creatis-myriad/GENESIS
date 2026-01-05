import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import summarize
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from torch import nn


class LightningModuleWrapper(pl.LightningModule):
    """Wraps a native PyTorch module inside a LightningModule to use Lightning's summary utilities."""

    def __init__(self, model: nn.Module) -> None:  # noqa: D107
        super().__init__()
        self.model = model


# Instantiate TabPFNClassifier and fit it on (any) data to fully create the model
tabpfn_clf = TabPFNClassifier()
X, y = load_breast_cancer(return_X_y=True)  # Used same dataset example as in TabPFN's readme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
tabpfn_clf.fit(X_train, y_train)

pl_wrapper = LightningModuleWrapper(tabpfn_clf.model_)
print(summarize(pl_wrapper, max_depth=-1))
