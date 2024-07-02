import snntorch as snn
import torch
import torch.nn as nn

class Net2_CSNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, betas, spike_grad, num_steps, thresholds):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.betas = betas
        self.spike_grad = spike_grad
        self.num_steps = num_steps
        self.thresholds = thresholds
        # Architecture
        self.architecture = "Conv1d(1, 32, 2) + MaxPool(2) + LIF + Conv1d(32, 128, 2) + MaxPool(2) + LIF + Conv1d(128, 256, 2) + MaxPool(2) + LIF + Linear(256, 2) + LIF"
        # Layer1
        self.conv1 = nn.Conv1d(1, 32, 2)
        self.mp1 = nn.MaxPool1d(2)
        self.lif1 = snn.Leaky(beta=self.betas[0], spike_grad=spike_grad, threshold=self.thresholds[0], learn_beta=True, learn_threshold=True)
        # Layer2
        self.conv2 = nn.Conv1d(32, 128, 2)
        self.mp2 = nn.MaxPool1d(2)
        self.lif2 = snn.Leaky(beta=self.betas[1], spike_grad=spike_grad, threshold=self.thresholds[1], learn_beta=True, learn_threshold=True)
        # Layer2
        self.conv3 = nn.Conv1d(128, 256, 2)
        self.mp3 = nn.MaxPool1d(2)
        self.lif3 = snn.Leaky(beta=self.betas[2], spike_grad=spike_grad, threshold=self.thresholds[2], learn_beta=True, learn_threshold=True)
        # Layer3
        self.fc1 = nn.Linear(768, 2) 
        self.lif4 = snn.Leaky(beta=self.betas[3], spike_grad=spike_grad, threshold=self.thresholds[3], learn_beta=True, learn_threshold=True, output=True)
        
    def forward(self, x):
        cur_last_rec = []
        spk_last_rec = []
        mem_last_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        for _ in range(self.num_steps):
            # Layer1
            cur1 = self.mp1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            # Layer2
            cur2 = self.mp2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            # Layer3
            cur3 = self.mp3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            # Layer4
            cur4 = self.fc1(spk3.flatten(1))    
            spk4, mem4 = self.lif4(cur4, mem4)
            # Record the final layer
            cur_last_rec.append(cur4)
            spk_last_rec.append(spk4)
            mem_last_rec.append(mem4)
        return torch.stack(cur_last_rec, dim=0), torch.stack(spk_last_rec, dim=0), torch.stack(mem_last_rec, dim=0)
        
    def get_architecture(self):
        return self.architecture
    
    def get_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    