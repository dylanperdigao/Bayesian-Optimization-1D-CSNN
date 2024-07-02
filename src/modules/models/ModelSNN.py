# imports
import numpy as np
import torch

from snntorch.surrogate import fast_sigmoid
from torch import from_numpy, nn
from torch.utils.data import DataLoader
from modules.dataset.DatasetBAF import DatasetBAF
from modules.networks.csnn import Net1_CSNN, Net2_CSNN, Net3_CSNN
from modules.metrics import evaluate, evaluate_aequitas

class ModelSNN(object):
    """ModelSNN class.
    ------------------------------------------------------
    Attributes:
        num_features (int): number of features
        num_classes (int): number of classes
        architecture (str): architecture of the network
        class_weights (tuple): class weights
        batch_size (int): batch size
        betas (tuple): betas for the decay of the membrane potential
        slope (int): slope for the spike grad
        thresholds (tuple): thresholds for the membrane potential
        num_epochs (int): number of epochs
        num_steps (int): number of steps
        adam_betas (tuple): betas for the Adam optimizer
        learning_rate (float): learning rate for the optimizer
        network (torch.nn.Module): network object
        gpu_number (int): number of the GPU
        verbose (int): verbosity level
        _dtype (torch.dtype): data type for the network
        _device (torch.device): device for the network
        _optimizer (torch.optim.Adam): optimizer object
    ------------------------------------------------------
    Methods:
        _load_data(x, y): Load the data into the network.
        _loadnetwork(): Load the network based on the architecture.
        fit(x_train, y_train): Training loop for the network.
        predict(x_test, y_test): Predict the labels for the test set.
        evaluate(targets, predicted): Evaluate the model using the confusion matrix and some metrics.
        evaluate_aequitas(x_test, y_test, predictions): Evaluate the model using the Aequitas library.
        get_parameters(): Get the parameters of the network.
        get_num_params(): Get the number of parameters of the network.
    """
    def __init__(self, num_features, num_classes, architecture, **kwargs):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                device = torch.device(f"cuda:{kwargs.get('gpu_number', 0)}")
            except Exception as e:
                print(f"Error while trying to select the GPU: {e}")
                device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        self.num_features = num_features
        self.num_classes = num_classes
        self.architecture = architecture
        self.batch_size = kwargs.get("batch_size", 32)
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.num_steps = kwargs.get("num_steps", 10)
        self.adam_betas = kwargs.get("adam_betas", (0.97, 0.999))
        self.verbose = kwargs.get("verbose", 0)
        self.betas = kwargs.get("betas", (0.75, 0.75, 0.75))
        self.slope = kwargs.get("slope", 25)
        self.thresholds = kwargs.get("thresholds", (1.0, 1.0, 1.0))
        self.spike_grad = fast_sigmoid(slope=self.slope)
        self._dtype = torch.float
        self._device = device
        self.network = self._loadnetwork()
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.class_weights = torch.tensor(kwargs.get("class_weights", (0.998, 0.002)), dtype=self._dtype, device=self._device)
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=self.adam_betas)
        self._loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        print_str = [
            "=======================================",
            "ModelSNN",
            "=======================================",
            f"- Device: {self._device}",
            f"- Batch size: {self.batch_size}",
            f"- Epochs: {self.num_epochs}",
            f"- Steps: {self.num_steps}",
            f"- Betas: {self.betas}",
            f"- Spike grad slope: {self.slope}",
            f"- Thresholds: {self.thresholds}",
            f"- Class weights: {self.class_weights}",
            f"- Adam betas: {self.adam_betas}",
            f"- Learning rate: {self.learning_rate}",
            "======================================="
        ]
        print("\n".join(print_str)) if self.verbose >= 1 else None

    
    def _load_data(self, x, y):
        """Load the data into the network.
        ------------------------------------------------------
        Args:
            x (pd.DataFrame): dataframe with the features
            y (pd.Series): series with the labels
        ------------------------------------------------------
        Returns:
            loader (torch.utils.data.DataLoader): data loader object
        """
        x_np = from_numpy(x.values).float().unsqueeze(1)
        y_np = from_numpy(y.values).int()
        ds = DatasetBAF(x_np, y_np)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return loader

    def _loadnetwork(self):
        """Load the network based on the architecture.
        ------------------------------------------------------
        Returns:
            network (torch.nn.Module): network object
        """
        if self.architecture == "Net1_CSNN":
            network = Net1_CSNN(self.num_features, self.num_classes, self.betas, self.spike_grad, self.num_steps, self.thresholds)
        elif self.architecture == "Net2_CSNN":
            network = Net2_CSNN(self.num_features, self.num_classes, self.betas, self.spike_grad, self.num_steps, self.thresholds)
        elif self.architecture == "Net3_CSNN":
            network = Net3_CSNN(self.num_features, self.num_classes, self.betas, self.spike_grad, self.num_steps, self.thresholds)
        else:
            raise ValueError(f"Architecture {self.architecture} not found.")
        print(network) if self.verbose >= 2 else None
        return network.to(self._device)

    def fit(self, x_train, y_train):
        """Training loop for the network.
        ------------------------------------------------------
        Args:
            x_train (pd.DataFrame): dataframe with the training features
            y_train (pd.Series): series with the training labels
        """
        self._train_loader = self._load_data(x_train, y_train)
        for epoch in range(self.num_epochs):
            print(f"Epoch - {epoch}") if self.verbose >= 2 else None
            train_batch = iter(self._train_loader)
            for data, targets in train_batch:
                data = data.to(self._device)
                targets = targets.to(self._device, dtype=torch.long)
                self.network.train()
                _, _, mem_rec = self.network(data) 
                loss_val = torch.zeros((1), dtype=self._dtype, device=self._device)
                for step in range(self.num_steps):
                    loss_val += self._loss_fn(mem_rec[step], targets)
                self._optimizer.zero_grad()
                loss_val.backward()
                self._optimizer.step()
                print(f"Loss: {loss_val.item()}") if self.verbose >= 3 else None
               


    def predict(self, x_test, y_test):
        """Predict the labels for the test set.
        ------------------------------------------------------
        Args:
            x_test (pd.DataFrame): dataframe with the test features
            y_test (pd.Series): series with the test labels
        ------------------------------------------------------
        Returns:
            predictions (np.array): array with the predicted labels
            test_targets (np.array): array with the true labels
        """
        print("Predicting...") if self.verbose >= 2 else None
        self._test_loader = self._load_data(x_test, y_test)
        predictions = np.array([])
        test_targets = np.array([])
        with torch.no_grad():
            self.network.eval()
            for data, targets in iter(self._test_loader):
                # Move data to device
                data = data.to(self._device)
                targets = targets.to(self._device, dtype=torch.long)
                # forward-pass
                _, spk_rec, _ = self.network(data)
                spike_count = spk_rec.sum(0)
                _, max_spike = spike_count.max(1)
                predictions = np.append(predictions, max_spike.cpu().numpy())
                test_targets = np.append(test_targets, targets.cpu().numpy())
        return predictions, test_targets
    

    def evaluate(self, targets, predicted):
        """Evaluate the model using the confusion matrix and some metrics.
        ------------------------------------------------------ 
        Args:
            targets (list): list of true values
            predicted (list): list of predicted values
        ------------------------------------------------------ 
        Returns:
            cm (np.array): confusion matrix
            accuracy (float): accuracy of the model
            precision (float): precision of the model
            recall (float): recall of the model
            fpr (float): false positive rate of the model
            f1_score (float): f1 score of the model
            auc (float): area under the curve of the model
        """
        metrics = evaluate(targets, predicted)
        print_str = [
            "=======================================",
            'Confusion Matrix:',
            f"{metrics['tp']}(TP)\t{metrics['fn']}(FN)",
            f"{metrics['fp']}(FP)\t{metrics['tn']}(TN)",
            "---------------------------------------",
            f'FPR:\t\t{metrics["fpr"]*100:.4f}%',
            f'Recall:\t\t{metrics["recall"]*100:.4f}%',
            f'TNR:\t\t{metrics["tnr"]*100:.4f}%',
            f'Accuracy:\t{metrics["accuracy"]*100:.4f}%',
            f'Precision:\t{metrics["precision"]*100:.4f}%',
            f'F1 Score:\t{metrics["f1_score"]*100:.4f}%',
            f'AUC:\t\t{metrics["auc"]*100:.4f}%',
            "=======================================",
        ]
        print("\n".join(print_str)) if self.verbose >= 1 and (metrics["recall"]>0.1 and metrics["fpr"]<0.1) else None
        return metrics
    
    def evaluate_aequitas(self, x_test, y_test, predictions):
        """Evaluate the model using the Aequitas library.
        ------------------------------------------------------
        Args:
            x_test (pd.DataFrame): dataframe with the test features
            y_test (pd.Series): series with the test labels
            predictions (np.array): array with the predictions
        ------------------------------------------------------
        Returns:
            threshold (float): threshold for the model
            fpr@5FPR (float): false positive rate of the model
            recall@5FPR (float): recall of the model
            tnr@5FPR (float): true negative rate of the model
            accuracy@5FPR (float): accuracy of the model
            precision@5FPR (float): precision of the model
            f1_score@5FPR (float): f1 score of the model
            fpr_ratio (float): false positive rate ratio of the model
            fnr_ratio (float): false negative rate ratio of the model
            recall_older (float): recall of the older group
            recall_younger (float): recall of the younger group
            fpr_older (float): false positive rate of the older group
            fpr_younger (float): false positive rate of the younger group
            fnr_older (float): false negative rate of the older group
            fnr_younger (float): false negative rate of the younger group
        """
        metrics = evaluate_aequitas(x_test, y_test, predictions)
        print_str = [
            "=======================================",
            'Aequitas Metrics:',
            '---------------------------------------',
            f'FPR@5FPR:\t{metrics["fpr@5FPR"]*100:.4f}%',
            f'Recall@5FPR:\t{metrics["recall@5FPR"]*100:.4f}%',
            f'TNR@5FPR:\t{metrics["tnr@5FPR"]*100:.4f}%',
            f'Accuracy@5FPR:\t{metrics["accuracy@5FPR"]*100:.4f}%',
            f'Precision@5FPR:\t{metrics["precision@5FPR"]*100:.4f}%',
            f'F1-Score@5FPR:\t{metrics["f1_score@5FPR"]*100:.4f}%',
            '---------------------------------------',
            f'FPR Ratio:\t{metrics["fpr_ratio"]*100:.4f}',
            f'FNR Ratio:\t{metrics["fnr_ratio"]*100:.4f}',
            '---------------------------------------',
            f'Threshold:\t{metrics["threshold"]}',
            f'Recall Older:\t{metrics["recall_older"]*100:.4f}%',
            f'Recall Younger:\t{metrics["recall_younger"]*100:.4f}%',
            f'FPR Older:\t{metrics["fpr_older"]*100:.4f}%',
            f'FPR Younger:\t{metrics["fpr_younger"]*100:.4f}%',
            f'FNR Older:\t{metrics["fnr_older"]*100:.4f}%',
            f'FNR Younger:\t{metrics["fnr_younger"]*100:.4f}%',
            "=======================================",
        ]
        print("\n".join(print_str)) if self.verbose >= 1 and (metrics["recall@5FPR"]>0.1 and metrics["fpr@5FPR"]<0.1) else None
        return metrics
    
    def get_parameters(self):
        """Get the parameters of the network."""
        return [p for p in self.network.parameters() if p.requires_grad]
    
    def get_num_params(self):
        """Get the number of parameters of the network."""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
