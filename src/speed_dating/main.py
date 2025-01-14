import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
from MLP_model import MLP

parser = argparse.ArgumentParser(description='SpeedDATING')
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--env', type=int, default=22)
# parser.add_argument('--env', type=int, default=0)
# parser.add_argument('--collider', type=int, default=1)
parser.add_argument('--collider', type=int, default=0)
parser.add_argument('--num_col', type=int, default=0)
parser.add_argument('--mod', type=str, default='mod4')
parser.add_argument('--dimension', type=str, default='high')
parser.add_argument('--dat', type=int, default=1)
parser.add_argument('--net', type=str, default='tarnet')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--icp', type=int, default=1)
parser.add_argument('--all_train', type=int, default=0)
# parser.add_argument('--all_train', type=int, default=1)
parser.add_argument('--data_base_dir', type=str, default='')
parser.add_argument('--output_base_dir', type=str, default='')

flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []

final_train_ate = []
final_test_ate = []
final_test_ate_error = []
final_train_ate_error = []
final_train_treatment_acc=[]
final_test_treatment_acc=[]

torch.manual_seed(1)

# Load data
file_path = flags.data_base_dir
df_path = os.path.join(file_path,
                       '{}/speedDate{}{}{}.csv'.format(flags.mod, flags.mod, flags.dimension, str(flags.dat)))
df = pd.read_csv(df_path)
data = df.values
oracle = pd.read_csv(
    file_path + '{}/speedDate{}{}Oracle{}.csv'.format(flags.mod, flags.mod, flags.dimension, str(flags.dat)))
ITE_oracle = oracle['ITE'].values.reshape(-1, 1)

reader = pd.read_csv(file_path + flags.mod + '/ate_truth.csv')
truth = {
    'low': reader['low_ate'].values,
    'med': reader['med_ate'].values,
    'high': reader['high_ate'].values
}

# print("data", data.shape) #(6000, 187)
# rows: different data, columns: Y, A(our T), 185 covariates

Y = data[:, 0].reshape(-1, 1) # (6000, 1)
T = data[:, 1].reshape(-1, 1) # (6000, 1)
X = data[:, 2:] # (6000, 185)

# turn one covariate into environments, and delete the covariate from our known information
# index = X[:, flags.env].argsort() # (6000, )
# X = np.delete(X, flags.env, axis=1)

def prepare_data(X, Y, T, ITE_oracle):
    if flags.gpu == 1:
        return {
            'covariates': torch.from_numpy(X).cuda(),
            'outcome': torch.from_numpy(Y).cuda(),
            'treatment': torch.from_numpy(T).cuda(),
            'ITE_oracle': torch.from_numpy(ITE_oracle)
        }
    else:
        return {
            'covariates': torch.from_numpy(X),
            'outcome': torch.from_numpy(Y),
            'treatment': torch.from_numpy(T),
            'ITE_oracle': torch.from_numpy(ITE_oracle)
        }

def split_data(X, Y, T, ITE_oracle, train_split = 0.75):
    data_num = data.shape[0]
    train_num = int(data_num*train_split)
    test_num = data_num - train_num
    # print("X[:train_num]: ", X[:train_num].shape) # (4500, 185)
    # print("X[train_num:]: ", X[train_num:].shape) # (1500, 185)
    train_env = prepare_data(X[:train_num], Y[:train_num], T[:train_num], ITE_oracle[:train_num])
    test_env = prepare_data(X[train_num:], Y[train_num:], T[train_num:], ITE_oracle[train_num:])
    return train_env, test_env

train_env, test_env = split_data(X, Y, T, ITE_oracle, train_split = 0.75)
dataloaders = {
    "train": train_env,
    "test": test_env
}


# create MLP model
mlp = MLP(X.shape[1], flags)
if flags.gpu==1:
    mlp.cuda()
else:
    mlp


# Define loss function helpers
def mean_nll(y_logit, y):
    return nn.functional.binary_cross_entropy_with_logits(y_logit, y.float())

def mean_accuracy(y_logit, y):
    # check correct binary -- for T and Y
    preds = (y_logit > 0.).double()
    return ((preds - y).abs() < 1e-2).float().mean()


def penalty(y_logit, y):
    if flags.gpu ==1:
        scale=torch.tensor(1.).cuda().requires_grad_()
    else:
        scale = torch.tensor(1.).requires_grad_()
    loss = mean_nll(y_logit * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    res = torch.sum(grad ** 2)
    return res


def ite(y0_logit, y1_logit):
    y0_pred = torch.sigmoid(y0_logit).float()
    y1_pred = torch.sigmoid(y1_logit).float()
    return y1_pred - y0_pred


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc', 'train_ate', 'train_ate_error', 'test_ate', 'test_ate_error', 'train_t_acc',
             'test_t_acc')

def train_dataloader(val, dataloaders, all_train=1):
    if all_train == 1:
        if dataloaders["train"][val].dim() == 0:
            return torch.stack((dataloaders["train"][val], dataloaders["test"][val])).mean()
        else: 
            # TODO: verify if this looks correct to get mean
            return torch.cat((dataloaders["train"][val], dataloaders["test"][val])).mean()
    else: # train_test split
        return dataloaders["train"][val].mean() 

#different choice of optimizer yields different performance.
optimizer_adam = optim.Adam(mlp.parameters(), lr=flags.lr)
optimizer_sgd = optim.SGD(mlp.parameters(), lr=1e-7, momentum=0.9)

# training loop
for step in range(flags.steps):
    for setting, single_env in dataloaders.items():
        # print("setting: ", setting)
        # print("single_env: ", single_env.keys())
        
        logits = mlp(single_env['covariates'].float())
        y0_logit = logits[:, 0].unsqueeze(1)
        y1_logit = logits[:, 1].unsqueeze(1)
        t_logit = logits[:, 2].unsqueeze(1)
        t = single_env['treatment'].float()
        # print("y0_logit: ", y0_logit.shape) # (1500, 1) # (4500, 1)
        # print("y1_logit: ", y1_logit.shape) # (1500, 1) # (4500, 1)
        # print("t_logit: ", t_logit.shape) # (1500, 1) # (4500, 1)
        # print("t: ", t.shape) # (1500, 1) # (4500, 1)
        y_logit = t * y1_logit + (1 - t) * y0_logit

        single_env['ite'] = ite(y0_logit, y1_logit)

        single_env['nll'] = mean_nll(y_logit, single_env['outcome'])
        single_env['t_nll'] = mean_nll(t_logit, t)

        # check if identify T and Y with correct 0 or 1 -- binary treatment, binary outcome
        single_env['acc'] = mean_accuracy(y_logit, single_env['outcome'])
        single_env['t_acc'] = mean_accuracy(t_logit, single_env['treatment'])
        # TODO: what is this additional penalty
        single_env['penalty'] = penalty(y_logit, single_env['outcome'])

    train_nll = train_dataloader('nll', dataloaders, all_train=flags.all_train)
    train_t_nll = train_dataloader('t_nll', dataloaders, all_train=flags.all_train)
    train_acc = train_dataloader('acc', dataloaders, all_train=flags.all_train)
    train_t_acc = train_dataloader('t_acc', dataloaders, all_train=flags.all_train)

    train_ate = train_dataloader('ite', dataloaders, all_train=flags.all_train)
    train_penalty = train_dataloader('penalty', dataloaders, all_train=flags.all_train)

    if flags.gpu == 1:
        weight_norm = torch.tensor(0.).cuda()
    else:
        weight_norm = torch.tensor(0.)
    for w in mlp.parameters():
        weight_norm += w.norm().pow(2)

    loss = train_nll.clone()

    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight
                      if step >= flags.penalty_anneal_iters else 1)
    loss += penalty_weight * train_penalty
    loss += train_t_nll

    if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight
    if step < 501:
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()
    #train with sgd after the performance is stablized with adam.
    else:
        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()
    
    test_acc = dataloaders["test"]['acc']
    test_t_acc = dataloaders["test"]['t_acc']
    test_ate = dataloaders["test"]['ite'].mean()

    pred_ite = dataloaders["test"]['ite'].detach().cpu().numpy()
    true_ite = dataloaders["test"]['ITE_oracle'].detach().cpu().numpy()

    test_ate_error = abs(test_ate - true_ite.mean())
    train_ate_error = abs(train_ate - dataloaders["train"]['ITE_oracle'].detach().cpu().numpy().mean())


    # test_acc = envs[2]['acc']
    # test_t_acc = envs[2]['t_acc']
    # test_ate = envs[2]['ite'].mean()

    # pred_ite = torch.stack([envs[0]['ite'], envs[1]['ite'], envs[2]['ite']]).detach().cpu().numpy()
    # true_ite = torch.stack([envs[0]['ITE_oracle'], envs[1]['ITE_oracle'], envs[2]['ITE_oracle']]).detach().cpu().numpy()

    if step % 100 == 0:
        pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy() + train_t_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            train_penalty.detach().cpu().numpy(),
            test_acc.detach().cpu().numpy(),
            np.mean(train_ate.detach().cpu().numpy()),
            train_ate_error.detach().cpu().numpy(),
            np.mean(test_ate.detach().cpu().numpy()),
            test_ate_error.detach().cpu().numpy(),
            train_t_acc.detach().cpu().numpy(),
            test_t_acc.detach().cpu().numpy(),

        )
final_time = time.time()

# converting the tensor to numpy arrays
final_train_accs.append(train_acc.detach().cpu().numpy())
final_test_accs.append(test_acc.detach().cpu().numpy())
final_train_treatment_acc.append(train_t_acc.detach().cpu().numpy())
final_test_treatment_acc.append(test_t_acc.detach().cpu().numpy())

final_train_ate.append(train_ate.detach().cpu().numpy())
final_test_ate.append(test_ate.detach().cpu().numpy())
final_test_ate_error.append(test_ate_error.detach().cpu().numpy())
final_train_ate_error.append(train_ate_error.detach().cpu().numpy())


# print('Final train acc (mean/std across restarts so far):')
# print(np.mean(final_train_accs), np.std(final_train_accs))
#
# print('Final test acc (mean/std across restarts so far):')
# print(np.mean(final_test_accs), np.std(final_test_accs))
#
# print('Final train ate mae is:')
# print(abs(np.mean(final_train_ate) - truth[flags.dimension]))
#
# print('Final test ate mae is:')
# print(abs(np.mean(final_test_ate) - truth[flags.dimension]))
#
# print("PEHE is {}, std is: {}".format(np.square(pred_ite - true_ite).mean(), abs(pred_ite - true_ite).std()))
# print("MAE for ate is ", (abs((pred_ite).mean() - truth[flags.dimension])))

saver = {
    'pred_ite': [pred_ite],
    'sample_ite': [true_ite],
    'Y': single_env['outcome'].detach().cpu().numpy(),
    'T': single_env['treatment'].detach().cpu().numpy(),
    # 'index': [index]
}
if flags.net=='tarnet':
    tmp = os.path.join(flags.output_base_dir, 'tarnet/')
elif flags.net=='dragon':
    tmp = os.path.join(flags.output_base_dir, 'dragon/')
elif flags.net == 'tarnet_single':
    tmp = os.path.join(flags.output_base_dir, 'tarnet_single/')

if flags.collider==1:
    tmp = os.path.join(tmp, 'collider/')
else:
    tmp = os.path.join(tmp, 'no_collider/')
if flags.all_train == 1:
    log_path = os.path.join(tmp, "all_train/")
else:
    log_path = os.path.join(tmp, "train_test/")

os.makedirs(log_path, exist_ok=True)

save_path = os.path.join(log_path, "{}/{}/".format(flags.mod, flags.dimension))
os.makedirs(save_path, exist_ok=True)

if flags.penalty_weight > 0:
    for num, output in enumerate([saver]):
        np.savez_compressed(os.path.join(save_path, "irm_ite_output_{}".format(str(flags.dat))), **output)
else:
    for num, output in enumerate([saver]):
        np.savez_compressed(os.path.join(save_path, "erm_ite_output_{}".format(str(flags.dat))), **output)

#
final_output = pd.DataFrame({
    'train_acc': final_train_accs,
    'test_acc': final_test_accs,
    'train_ate': final_train_ate,
    'test_ate': final_test_ate,
    'train_treatment_acc': final_train_treatment_acc,
    'test_treatment_acc': final_test_treatment_acc,
    'PEHE': np.square(pred_ite - true_ite).mean(),
    'final_test_ate_error': final_test_ate_error,
    'final_train_ate_error': final_train_ate_error
})


if flags.penalty_weight > 0:
    tmp = save_path+ 'irm_ate_output_{}.csv'.format(str(flags.dat))
else:
    tmp = save_path + 'erm_ate_output_{}.csv'.format( str(flags.dat))

final_output.to_csv(tmp, index=False)