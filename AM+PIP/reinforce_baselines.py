import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from train import rollout, get_inner_model

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):

        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        # return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class SubsetDataset(Dataset):
    """Materialized contiguous slice of a dataset. Items are cloned so that pickling
    (baseline state in the epoch checkpoints) does not serialize the full source
    dataset's storage along with the tensor views."""

    def __init__(self, dataset, offset, num_samples):
        self.items = [
            {k: v.clone() for k, v in item.items()} if isinstance(item, dict) else item.clone()
            for item in (dataset[i] for i in range(offset, min(offset + num_samples, len(dataset))))
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, opts, epoch=0):
        super(Baseline, self).__init__()

        self.problem = problem
        self.opts = opts

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0] if not isinstance(dataset[0], dict) else dataset[0]['loc']).size(0) != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            if getattr(self.opts, 'train_set_path', None) is not None:
                # fixed-dataset training: draw the baseline evaluation dataset from the same fixed
                # instance distribution (random slice of the training file) instead of the generator.
                # The file is read once per process and cached on opts (shared with train_epoch);
                # later baseline updates slice the in-memory copy, so training does not depend on
                # the data file staying available on disk for the whole run.
                if getattr(self.opts, '_fixed_train_dataset', None) is None:
                    self.opts._fixed_train_dataset = self.problem.make_dataset(
                        filename=self.opts.train_set_path, size=self.opts.graph_size,
                        num_samples=int(1e9), hardness=self.opts.hardness)
                    print(">> Loaded fixed dataset: {} ({} instances)".format(
                        self.opts.train_set_path, len(self.opts._fixed_train_dataset)))
                fixed = self.opts._fixed_train_dataset
                offset = torch.randint(0, max(len(fixed) - self.opts.val_size, 1), (1,)).item()
                self.dataset = SubsetDataset(fixed, offset, self.opts.val_size)
            else:
                self.dataset = self.problem.make_dataset(
                    size=self.opts.graph_size, num_samples=self.opts.val_size, hardness=self.opts.hardness)
        else:
            self.dataset = dataset
        print(">> Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        if len(self.bl_vals.shape) > 1:
            self.bl_vals = self.bl_vals.sum(-1)
            # if (self.bl_vals[:,-1] == 0).any(): # have feasible
            #     self.mean = self.bl_vals[self.bl_vals[:,-1] == 0].mean()
            # else: # no feasible
            #     self.mean = self.bl_vals.sum(-1).mean()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print(">> Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print(">> Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
        if len(candidate_vals.shape) > 1:
            candidate_vals = candidate_vals.sum(-1)
            # if (candidate_vals[:,-1] == 0).any(): # have feasible
            #     candidate_mean = candidate_vals[candidate_vals[:,-1] == 0].mean()
            # else: # no feasible
            #     candidate_mean = candidate_vals.sum(-1).mean()
        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print('>> Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        saved_model = state_dict['model']
        if isinstance(saved_model, list):
            # PIP-D runs wrap the model as [am_model, pip_model] (run.py), so a baseline
            # updated after that wrapping was checkpointed as a list; the baseline weights
            # are the AM half. The pip model is restored separately via --pip_checkpoint.
            saved_model = saved_model[0]
        get_inner_model(load_model).load_state_dict(get_inner_model(saved_model).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)
