import torch
import pickle
from torch import nn, no_grad, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset
import random
from pathlib import Path
import os

def convert_to_pretrained_path(old_path: str) -> str:
    parts = Path(old_path).parts

    task = parts[2]
    model_name = parts[3].replace("cityflow_", "")
    map_name = parts[4]
    gat_model = parts[7]
    setting = parts[8]

    new_path = Path("pretrained") / task / model_name / map_name / gat_model / setting
    return new_path.as_posix()





class BasePredictor(object):
    model_prefix = ""

    def __init__(self, rank, logger, DEVICE, model_dir, data_dir, backward=False, history=1):
        super().__init__()
        self.rank = rank
        self.epo = 0
        self.logger = logger
        self.DEVICE = DEVICE
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.backward = backward
        self.history = history
        self.batch_size = 64
        self.model = self.make_model().to(self.DEVICE).float()
        if not backward:
            self.criterion = nn.MSELoss()
            self.learning_rate = 0.0001
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.learning_rate = 0.00001
        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99)
        )
        self.online_optimizer = optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99)
        )
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def _direction_text(self):
        return "inverse" if self.backward else "forward"

    def _model_filename(self, e=""):
        txt = self._direction_text()
        if e is not None:
            return os.path.join(
                self.model_dir, f"{e}_{self.model_prefix}_inference_{txt}_{self.rank}.pt"
            )
        return os.path.join(
            convert_to_pretrained_path(self.model_dir),
            f"{self.model_prefix}_inference_{txt}_{self.rank}.pt",
        )

    def load_model(self, e=""):
        self.model = self.make_model()
        self.model.load_state_dict(torch.load(self._model_filename(e)))
        self.model = self.model.float().to(self.DEVICE)

    def save_model(self, e=""):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), self._model_filename(e))

    def make_model(self):
        raise NotImplementedError


class BaseNNPredictor(BasePredictor):
    model_prefix = "NN"

    def __init__(
        self,
        rank,
        logger,
        state_dim,
        action_dim,
        out_dim,
        DEVICE,
        model_dir,
        data_dir,
        backward=False,
        history=1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.out_dim = out_dim
        super().__init__(rank, logger, DEVICE, model_dir, data_dir, backward, history)

    def predict(self, state, action):
        state, action = state.to(self.DEVICE), action.to(self.DEVICE)
        with torch.no_grad():
            return self.model(state, action)

    def get_train_dataset_path(self, agent_num=None):
        raise NotImplementedError

    def get_test_dataset_path(self, agent_num=None):
        raise NotImplementedError

    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError

    def _run_eval(self, dataset, e, txt):
        test_loss = 0.0
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for state, action, y_true in test_loader:
                state = state.to(self.DEVICE, non_blocking=True)
                action = action.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)
                y_pred = self.model(state, action)
                test_loss += self.compute_loss(y_pred, y_true).item()
        test_loss = test_loss / len(dataset)
        self.logger.info(f"epoch {e}: {txt} test average loss {test_loss}.")
        self.model.train()
        return test_loss

    def train(self, epochs, sign, agent_num=None, max_samples=5000, mode=None, jlnet=0):
        train_loss = 0.0
        full_dataset = PKLDataset(self.get_train_dataset_path(agent_num))
        subset_size = min(max_samples, len(full_dataset)) if max_samples else len(full_dataset)
        subset_indices = random.sample(range(len(full_dataset)), subset_size)
        train_dataset = Subset(full_dataset, subset_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        txt = self._direction_text()

        if agent_num is not None:
            self.logger.info(f"{txt} model: training agent {agent_num}.")
        else:
            print(f"Epoch {self.epo - 1} Training")

        self.model.to(self.DEVICE)
        for e in range(epochs):
            for state, action, y_true in train_loader:
                state = state.to(self.DEVICE, non_blocking=True)
                action = action.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                y_pred = self.model(state, action)
                loss = self.compute_loss(y_pred, y_true)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if e == 0 or e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f"Epoch {e}: {txt} train average loss {ave_loss}.")
                self._run_eval(PKLDataset(self.get_test_dataset_path(agent_num)), e, txt)
            train_loss = 0.0

        self.epo += 1
        if sign == "inverse":
            return 0


class NN_predictor(BaseNNPredictor):
    def make_model(self):
        return N_net(self.state_dim, self.action_dim, self.out_dim, self.backward).float()

    def get_train_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for NN_predictor training.")
        return f"collected/ereal_train_full_agent_{agent_num}.pkl"

    def get_test_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for NN_predictor evaluation.")
        return f"collected/ereal_test_full_agent_{agent_num}.pkl"

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true.squeeze(1))


# Custom Dataset class to load the .pkl file
class PKLDataset(Dataset):
    def __init__(self, pkl_file, flag=''):
        self.flag = flag
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)  # Load the (features, targets) list from the .pkl file
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):

        if self.flag == 'jlg':
            states, n_states, n_actions, pred_state, targets = self.data[idx]
            return (
                torch.as_tensor(states, dtype=torch.float32),
                torch.as_tensor(n_states, dtype=torch.float32),
                torch.as_tensor(n_actions, dtype=torch.float32),
                torch.as_tensor(pred_state, dtype=torch.float32),
                torch.as_tensor(targets, dtype=torch.float32)
            )
        
        elif self.flag == 'central':
            states, next_states, targets = self.data[idx]
            return (
                torch.as_tensor(states, dtype=torch.float32),
                torch.as_tensor(next_states, dtype=torch.float32),
                torch.as_tensor(targets, dtype=torch.long)
            )
        else:
            states, actions, targets = self.data[idx]
            return (
                torch.as_tensor(states, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(targets, dtype=torch.float32)
            )

import torch
import torch.nn as nn
import torch.nn.functional as F

class N_net(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(N_net, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Per-agent processing after concatenation
        self.agent_fc = nn.Linear(128, 128)  # 64 (state) + 64 (action)
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(state_dim[0] * 128, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        # Concatenate state and action per agent
        x = torch.cat((state_out, action_out), dim=-1)  # (batch, num_agents, 128)
        
        # Process per-agent embeddings
        x = F.relu(self.agent_fc(x))  # (batch, num_agents, 128)
        
        # Flatten across agents
        x = x.view(batch_size, -1)  # (batch, num_agents * 128)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = self.dense_5(x)
        
        return x



class N_net_noAction(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(N_net_noAction, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(state_dim[0] * 64 + 64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        x = torch.cat((state_out, action_out), dim=1)  # (batch, num_agents, 128)

        x = x.view(state_out.shape[0], 1, -1)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = self.dense_5(x)
        
        return x



class N_net_noState(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(N_net_noState, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(action_dim[0] * 64 + 64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        x = torch.cat((state_out, action_out), dim=1)  # (batch, num_agents, 128)

        x = x.view(state_out.shape[0], 1, -1)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = self.dense_5(x)
        
        return x


class Central_N_net(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(Central_N_net, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Per-agent processing after concatenation
        self.agent_fc = nn.Linear(128, 128)  # 64 (state) + 64 (action)
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(state_dim[0] * 128, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 192)
        self.dense_5 = nn.Linear(int(192 / state_dim[0]), size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        # Concatenate state and action per agent
        x = torch.cat((state_out, action_out), dim=-1)  # (batch, num_agents, 128)
        
        # Process per-agent embeddings
        x = F.relu(self.agent_fc(x))  # (batch, num_agents, 128)

        # Concatenate all states and actions
        x = x.view(batch_size, 1, -1)  # (batch, 1, -1)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))

        # Reshape output back to per-agent next state
        x = x.view(batch_size, num_agents, -1)  # (batch, num_agents, -1)
        
        x = self.dense_5(x)

        return x


class BaseUncertaintyPredictor(BasePredictor):
    model_prefix = "UNCERTAINTY"

    def __init__(
        self,
        rank,
        logger,
        ind_state_dim,
        n_state_dim,
        action_dim,
        pred_state_dim,
        out_dim,
        DEVICE,
        model_dir,
        data_dir,
        backward=False,
        history=1,
    ):
        self.ind_state_dim = ind_state_dim
        self.n_state_dim = n_state_dim
        self.action_dim = action_dim
        self.pred_state_dim = pred_state_dim
        self.out_dim = out_dim
        super().__init__(rank, logger, DEVICE, model_dir, data_dir, backward, history)

    def predict(self, *model_inputs):
        model_inputs = [tensor.to(self.DEVICE) for tensor in model_inputs]
        with no_grad():
            result, uncertainty = self.model(*model_inputs)
        return result, uncertainty

    def get_train_dataset_path(self, agent_num=None):
        raise NotImplementedError

    def get_test_dataset_path(self, agent_num=None):
        raise NotImplementedError

    def build_dataset(self, dataset_path):
        raise NotImplementedError

    def unpack_batch(self, data):
        raise NotImplementedError

    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError

    def _run_eval(self, dataset, e, txt):
        test_loss = 0.0
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                model_inputs, y_true = self.unpack_batch(data)
                y_pred, _uncertainty = self.model(*model_inputs)
                test_loss += self.compute_loss(y_pred, y_true).item()
        test_loss = test_loss / len(dataset)
        self.logger.info(f"epoch {e}: {txt} test average loss {test_loss}.")
        self.model.train()
        return test_loss

    def train(self, epochs, sign, agent_num=None, max_samples=5000, mode=None, jlnet=0):
        train_loss = 0.0
        full_dataset = self.build_dataset(self.get_train_dataset_path(agent_num))
        subset_size = min(max_samples, len(full_dataset)) if max_samples else len(full_dataset)
        subset_indices = random.sample(range(len(full_dataset)), subset_size)
        train_dataset = Subset(full_dataset, subset_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        txt = self._direction_text()
        uncertainty = None

        if agent_num is not None:
            self.logger.info(f"{txt} model, training agent {agent_num}.")
        else:
            print(f"Epoch {self.epo - 1} Training")

        self.model.to(self.DEVICE)
        for e in range(epochs):
            for data in train_loader:
                model_inputs, y_true = self.unpack_batch(data)
                self.optimizer.zero_grad()
                y_pred, uncertainty = self.model(*model_inputs)
                loss = self.compute_loss(y_pred, y_true)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if e == 0 or e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f"epoch {e}: {txt} train average loss {ave_loss}.")
                self._run_eval(self.build_dataset(self.get_test_dataset_path(agent_num)), e, txt)
            train_loss = 0.0

        self.epo += 1
        if sign == "inverse":
            return uncertainty


class UNCERTAINTY_predictor(BaseUncertaintyPredictor):
    def make_model(self):
        return Inverse_N_net(
            self.ind_state_dim,
            self.n_state_dim,
            self.action_dim,
            self.pred_state_dim,
            self.out_dim,
            self.backward,
        ).float()

    def get_train_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for UNCERTAINTY_predictor training.")
        return f"collected/esim_train_full_agent_{agent_num}.pkl"

    def get_test_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for UNCERTAINTY_predictor evaluation.")
        return f"collected/esim_test_full_agent_{agent_num}.pkl"

    def build_dataset(self, dataset_path):
        return PKLDataset(dataset_path, "jlg")

    def unpack_batch(self, data):
        state, n_state, n_action, pred_state, y_true = data
        return (
            (
                state.to(self.DEVICE, non_blocking=True),
                n_state.to(self.DEVICE, non_blocking=True),
                n_action.to(self.DEVICE, non_blocking=True),
                pred_state.to(self.DEVICE, non_blocking=True),
            ),
            y_true.to(self.DEVICE, non_blocking=True),
        )

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true.squeeze().long())



class Inverse_N_net(nn.Module):
    def __init__(self, ind_state_dim, n_state_dim, n_action_dim, pred_state_dim, size_out, backward):
        super(Inverse_N_net, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)

        # Shared encoders for each neighbor
        self.n_state_encoder = nn.Linear(n_state_dim[-1], 64)
        self.n_action_encoder = nn.Linear(n_action_dim[-1], 64)
        
        # Encode concatenated neighbor states and actions
        self.neighbor_encoder = nn.Linear(128, 128)

        # Final concatenation: 64 (state) + 64 (pred_state) + 64 (aggregated neighbors) = 192
        self.dense_1 = nn.Linear(256 + 128 * n_state_dim[0], 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)


    def forward(self, state, n_state, n_action, pred_state):
            
        # Infer the number of neighbors dynamically
        num_neighbors = n_state.shape[1]

        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))

        # Encode each neighbor's state and action separately
        n_state_out = F.relu(self.n_state_encoder(n_state))  # Shape: [batch, num_neighbors, 64]
        n_action_out = F.relu(self.n_action_encoder(n_action))  # Shape: [batch, num_neighbors, 64]

        # Concatenate each neighbor's encoded state and action, then process
        neighbor_rep = F.relu(self.neighbor_encoder(torch.cat((n_state_out, n_action_out), dim=2)))  # [batch, num_neighbors, 64]
        
        batch_size = neighbor_rep.shape[0]

        # Concatenate with individual state and predicted state
        x = torch.cat((state_out, pred_state_out, neighbor_rep.view(batch_size, 1, -1)), dim=2)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        logits = self.EDL_layer(x).squeeze(1)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u


class Dec_Inverse_N_net(nn.Module):
    def __init__(self, ind_state_dim, pred_state_dim, size_out, backward):
        super(Dec_Inverse_N_net, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)
        
        # Final concatenation: 128 (state) + 128 (pred_state) = 256
        self.dense_1 = nn.Linear(ind_state_dim[0] * 128 + 128, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)
    
    def forward(self, ind_state, pred_state):
        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(ind_state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))

        # Concatenate encoded states
        x = torch.cat((state_out.view(state_out.shape[0], 1, -1), pred_state_out), dim=-1)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        logits = self.EDL_layer(x)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u



class Inverse_N_Net_No_State(nn.Module):
    def __init__(self, ind_state_dim, n_action_dim, pred_state_dim, size_out, backward):
        super(Inverse_N_Net_No_State, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)
        self.n_action_encoder = nn.Linear(n_action_dim[-1], 128)

        # Encode concatenated neighbor states and actions
        self.state_predAction = nn.Linear(128, 128)
    
        self.dense_1 = nn.Linear(256 + 128 * n_action_dim[0], 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)

    def forward(self, state, n_action, pred_state):
        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))
    
        # Encode each neighbor's action separately
        n_action_out = F.relu(self.n_action_encoder(n_action))
    
        # Concatenate along dimension 1 (feature dimension)
        x = F.relu(self.state_predAction(torch.cat((n_action_out, pred_state_out), dim=1)))
        
        x = x.view(pred_state_out.shape[0], 1, -1)

        x = torch.cat((x, state_out), dim=-1)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        logits = self.EDL_layer(x)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u



class Central_Inverse_N_net(nn.Module):
    def __init__(self, ind_state_dim, pred_state_dim, size_out, backward):
        super(Central_Inverse_N_net, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)

        # Per-agent processing after concatenation
        self.agent_fc = nn.Linear(256, 128)
        
        # Final concatenation: 128 (state) + 128 (pred_state) = 256
        self.dense_1 = nn.Linear(ind_state_dim[0] * 128, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 192)
        self.dense_4 = nn.Linear(int(192 / ind_state_dim[0]), 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)
    
    def forward(self, ind_state, pred_state):
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = ind_state.shape[:2]
        
        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(ind_state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))

        # Concatenate state and action per agent
        x = torch.cat((state_out, pred_state_out), dim=-1)  # (batch, num_agents, 256)

        x = F.relu(self.agent_fc(x))

        # Concatenate all states and actions
        x = x.view(batch_size, 1, -1)  # (batch, 1, -1)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        # Reshape output back to per-agent next state
        x = x.view(batch_size, num_agents, -1)  # (batch, num_agents, -1)
        
        x = F.relu(self.dense_4(x))

        logits = self.EDL_layer(x)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u
