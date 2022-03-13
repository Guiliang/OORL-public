import math
import torch
import numpy as np


def to_pt(np_matrix, enable_cuda=False, type='long'):
    """
    transfer numpy array to torch tensor
    """
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor))


def to_np(x):
    """
    transfer torch tensor to numpy array
    :param x: torch.tensor()
    :return:
    """
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    length = x.size(1)
    channels = x.size(2)
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return x + (signal.cuda() if x.is_cuda else signal)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = torch.nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


def masked_mean(x, m=None, dim=1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    x = x * m.unsqueeze(-1)
    mask_sum = torch.sum(m, dim=-1)  # batch
    tmp = torch.eq(mask_sum, 0).float()
    if x.is_cuda:
        tmp = tmp.cuda()
    mask_sum = mask_sum + tmp
    res = torch.sum(x, dim=dim)  # batch x h
    res = res / mask_sum.unsqueeze(-1)
    return res


def kl_divergence(mu_q, logvar_q, mu_p=None, logvar_p=None):
    if mu_p is None and logvar_p is None:
        kld = -0.5 * (1 + logvar_q - mu_q ** 2 - logvar_q.exp()).sum(1)
    else:
        kld = 0.5 * (logvar_p - logvar_q +
                     (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 1).sum(1)
    return kld


def load_graph_extractor(extractor, log_file, difficulty_level=None):
    """load the object extractor"""
    extractor_name_dict = {  # These are the named of trained extractor for different difficulty level
        1: 'df-1_sample-100_Apr-02-2021',
        3: 'df-3_sample-100_weight-0_May-12-2021',
        5: 'df-5_sample-100_weight-0_May-05-2021',
        7: 'df-7_sample-100_weight-0_May-10-2021',
        9: 'df-9_sample-100_Apr-12-2021',
        'mixed': 'df-general_Mar-23-2021',
        'general': 'df-general_Mar-23-2021'
    }
    if difficulty_level is None:
        difficulty_level = extractor.difficulty_level
    agent_load_path = extractor.output_dir + extractor.experiment_tag + "/saved_model_dynamic_{3}_{0}_{4}_{6}_{2}.pt".format(
        extractor.model.graph_decoding_method,
        extractor.graph_type,
        extractor_name_dict[difficulty_level],
        extractor.model.dynamic_model_type,
        extractor.model.dynamic_model_mechanism,
        difficulty_level,
        extractor.model.dynamic_loss_type,
        extractor.sample_number
    )
    extractor.load_pretrained_model(agent_load_path, log_file=log_file, load_partial_graph=False)


class HistoryScoreCache:

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


def memory_usage_psutil():
    # return the memory usage in MB
    import psutil, os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def ez_gather_dim_1(input, index):
    if len(input.size()) == len(index.size()):
        return input.gather(1, index)
    res = []
    for i in range(input.size(0)):
        res.append(input[i][index[i]])
    return torch.stack(res, 0)


def compute_mask(x):
    mask = torch.ne(x, 0).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.schedule = np.linspace(initial_p, final_p, schedule_timesteps)

    def value(self, step):
        if step < 0:
            return self.initial_p
        if step >= self.schedule_timesteps:
            return self.final_p
        else:
            return self.schedule[step]
