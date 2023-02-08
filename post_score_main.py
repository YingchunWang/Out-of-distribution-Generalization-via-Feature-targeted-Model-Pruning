import argparse
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import re
import torch.nn.functional as F
from torch import nn, optim, autograd
import torch.utils.data as Data
import torchvision
from dataset import MNIST_colored
from maskvgg import maskvgg11
from torch.nn.utils import clip_grad_norm_
import  cv2
import os
import logging
import math
from logging import FileHandler
from logging import StreamHandler
from dataset import prepare_ood_dataset
import torch.optim.lr_scheduler as lr_scheduler
from common import *
import os
import wandb
from dataset import manual_seed
from prun import prune_while_training


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def get_gpu_memory():
  free_gpu_info = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read()
  tep = re.findall('.*: (.*) MiB', free_gpu_info)
  gpu_dict = {}
  for one in range(len(tep)):
    gpu_dict[one] = int(tep[one])
  gpu_id = sorted(gpu_dict.items(), key=lambda item: item[1])[-1][0]
  os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(gpu_id))

# get_gpu_memory()
use_cuda = torch.cuda.is_available()
manual_seed(0)
parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--scheduler', type=list, default=[10, 15])
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--fine_tune_epochs', type=int, default=121)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=91257.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--fine_tune', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--train_set_size', type=int, default=50000)
parser.add_argument('--eval_interval', type=int, default=1)
parser.add_argument('--print_eval_intervals', type=str2bool, default=True)
parser.add_argument('--polar', type=bool, default=True)
parser.add_argument('--train_env_1__color_noise', type=float, default=0.0)
parser.add_argument('--train_env_2__color_noise', type=float, default=0.0)
#parser.add_argument('--val_env__color_noise', type=float, default=0.1)
parser.add_argument('--test_env__color_noise', type=float, default=0.0)

parser.add_argument('--erm_amount', type=float, default=1.0)

parser.add_argument('--early_loss_mean', type=str2bool, default=True)

parser.add_argument('--rex', type=str2bool, default=False)
parser.add_argument('--mse', type=str2bool, default=False)
parser.add_argument('--cox', type=str2bool, default=False)
parser.add_argument('--irm', type=str2bool, default=False)
parser.add_argument('--dro', type=str2bool, default=False)

parser.add_argument('--sim', type=str2bool, default=True)
parser.add_argument('--bn', type=str2bool, default=False)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--plot', type=str2bool, default=False)
parser.add_argument('--save_numpy_log', type=str2bool, default=False)
parser.add_argument('--thre_init', type=float, default=-10000.0)
parser.add_argument('--thre_cls', type=float, default=0.8)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--sparse', type=float, default=0.1)
parser.add_argument('--lambda_lasso', type=float, default=0.1,help='group lasso loss weight')

parser.add_argument('--target_remain_rate', type=float, default=0.65)
parser.add_argument('--logpath', type=str, default='nslim.txt')
parser.add_argument('--savepath', type=str, default='nslim_sparse.pth.tar')
parser.add_argument('--pruned_savepath', type=str, default='nslim_pruned.pth.tar')
#parser.add_argument('--resume', type=str, default="/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/checkpoint/full_2.pth.tar")
parser.add_argument('--resume', type=str, default=None)
args = parser.parse_args()
if (args.bn == False) and (args.cox == False):
  args.logpath = 'baseline.txt'
  args.savepath = 'baseline_sparse.pth.tar'
  args.savepath = 'baseline_pruned.pth.tar'
elif (args.bn == False) and (args.cox == True):
  args.logpath = 'cox.txt'
  args.savepath = 'cox_sparse.pth.tar'
  args.savepath = 'cox_pruned.pth.tar'
elif (args.bn == True) and (args.cox == False):
  args.logpath = 'normal.txt'
  args.savepath = 'normal_sparse.pth.tar'
  args.savepath = 'normal_pruned.pth.tar'


data_list = ['mnist', 'colored_object', 'scene_object']

data_set = data_list[2]

if data_set == 'mnist':
  args.batch_size = 5000
  cfg = None
else:
  args.epochs = 150
  args.scheduler = [10, 20, 50, 75, 100]
  args.batch_size =128
  args.lr = 0.0002
  cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512]

if args.irm == True:
  last_name = 'IRM'
elif args.rex == True:
  last_name = 'REX'
elif args.dro == True:
  last_name = 'DRO'
else:
  last_name = ''
if args.bn == True:
  first_name = 'MOD'
elif args.cox == True:
  first_name = 'SFP'
else:
  first_name = ''
name = first_name+last_name
if name =='':
  name = 'ERM'

root='./check_point'
logger_file = os.path.join(root,args.logpath)
logger=logging.getLogger()
logger.setLevel(logging.INFO) 
# Create FileHandler, output to file
log_file = logger_file
file_handler = logging.FileHandler(log_file, mode='w')
# Set lowest log level of this Handler
file_handler.setLevel(logging.INFO)
# set format
log_formatter = logging.Formatter('%(asctime)s[%(levelname)s]: %(message)s')
file_handler.setFormatter(log_formatter)
# create StreamHandler,output log to Stream
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

for k,v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))
  logging.info("\t{}: {}".format(k, v))

if args.gpu != None: 
    torch.cuda.set_device(args.gpu)
# TODO: logging
all_train_nlls = -1*np.ones((args.epochs, args.steps))
all_train_accs = -1*np.ones((args.epochs, args.steps))
#all_train_penalties = -1*np.ones((args.epochs, args.steps))
all_irmv1_penalties = -1*np.ones((args.epochs, args.steps))
all_rex_penalties = -1*np.ones((args.epochs, args.steps))
all_test_accs = -1*np.ones((args.epochs, args.steps))
all_grayscale_test_accs = -1*np.ones((args.epochs, args.steps))

final_train_accs = []
final_test_accs = []
highest_test_accs = []

class AverageMeter(object):

  def __init__(self):
    self.reset()
  
  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0
  
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

# additional subgradient descent on the sparsity-induced penalty term
def updateSaliencyBlock(lbd,mlp):  
  for n,p in mlp.named_parameters():
    if 'fc' in n:
      p.grad.data.add_(lbd * torch.sign(p.data))

    
def freeze_mask(model):
  for n,p in model.named_parameters():
    if 'mb' in n:
      p.required_grad = False
def mean_nll(logits, y):       
  loss = criterion(logits, y.squeeze())
  return loss  

def mean_accuracy(logits, y):
  """Computes the precision@k for the specified values of k"""
  output = logits.data
  batch_size = args.batch_size
  _, pred = output.topk(1, 1, True, True)
  pred = pred.t()
  correct = pred.eq(y.view(1, -1).expand_as(pred))
  top1 = correct[:1].contiguous().view(-1).float().sum(0)
  top1 = top1.mul_(100.0 /args.batch_size)
  return top1

def class_accuracy(logits, y):
  output = logits.data
  _, pred = output.topk(1, 1, True, True)
  pred = pred.t()
  correct = pred[0] == y
  class_acc = []
  for i in range(10):
    class_acc.append(100*((correct * (y==i)).sum())/((y==i).sum()))
  return class_acc

def penalty(logits, y):
  if use_cuda:
    scale = torch.tensor(1.).cuda().requires_grad_()
  else:
    scale = torch.tensor(1.).requires_grad_()
  
  loss = mean_nll(logits * scale, y.squeeze())
  grad = autograd.grad([loss.mean()], [scale], create_graph=True)[0]
  return torch.sum(grad**2)
# Train loop
def pretty_print(*values):
  col_width = 13
  def format_val(v):
    if not isinstance(v, str):
      v = np.array2string(v, precision=5, floatmode='fixed')
    return v.ljust(col_width)
  str_values = [format_val(v) for v in values]
  print("   ".join(str_values))

color_dict = {'0': [255,0,0], '1': [255,255,0], '2': [0,255,0], '3': [0,100,0], '4': [0,0,255], '5': [255, 0,255],'6': [0,0,128], '7': [220,220,220], '8': [255,255,255], '9': [0,255,255]}
num_classes = 10
def create_env(p, val=False, batch_size = 5000, dataset = 'mnist'):
  # if os.path.exists('Mixed_Mnist_train_{}.pt'.format(p)):
  #   pass
  # else:
  mixed_train, mixed_test, original_test_set, colored_test_set = prepare_ood_dataset(dataset, p)
  if val:
    loader_id = torch.utils.data.DataLoader(colored_test_set, len(mixed_test), shuffle=True)
    loader_ood = torch.utils.data.DataLoader(original_test_set, len(mixed_test), shuffle=True)
    loader = torch.utils.data.DataLoader(mixed_test, len(mixed_test), shuffle=True)
    return {
      "loader": loader,
      "loader_id": loader_id,
      "loader_ood": loader_ood
    }
  else:
    loader = torch.utils.data.DataLoader(mixed_train, batch_size, shuffle=True)
    return {
      "loader": loader
    }

def make_environment(images, labels, e):
  def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
  def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1
 
  images = images.reshape((-1, 28, 28))
  #images = np.array([cv2.resize(images[i].detach().cpu().numpy().astype(np.uint8),(48,48)) for i in range(images.shape[0])])
  #images = torch.tensor(images.reshape(-1,48,48)).cuda()

  images = torch.stack([images, images, images], dim=3)
        
  # Assign a color based on the label; flip the color with probability e
  for img in range(len(images)):
      args = torch_bernoulli(e, len(labels))
      label = labels[img]
      
      if args[img] > 0:
          color = color_dict[str(int(np.array(label)))]
      else:
          color = color_dict[str(np.array(torch.randint(10,[1,]))[0])]
      for rgb in range(3):
          c_color = color[rgb]
          
          images[img, :, :, rgb] *= c_color
  images =  images.reshape(-1,3,28,28)[:, :, ::2, ::2]

  all_image = (images.float() / 255.).cuda()
  all_label = labels[:, None].cuda()
  return {
    'images': (images.float() / 255.).cuda(),
    'labels': labels[:, None].cuda()
  }
# train_data = torchvision.datasets.MNIST(root='/data/wyc', train=True,
#                                         transform=torchvision.transforms.ToTensor(),
#                                         download=True)
# val_data = torchvision.datasets.MNIST(root='/data/wyc',
#                                        transform=torchvision.transforms.ToTensor(),
#                                        train=False)
# test_data = torchvision.datasets.MNIST(root='/data/wyc',
#                                        transform=torchvision.transforms.ToTensor(),
#                                        train=False)
# mnist_train = (train_data.data[:50000], train_data.targets[:50000])
# mnist_val = (train_data.data[40000:60000], train_data.targets[40000:60000])
# mnist_test = (test_data.data[:10000], test_data.targets[:10000])
# rng_state = np.random.get_state()
# np.random.shuffle(mnist_train[0].numpy())
# np.random.set_state(rng_state)
# np.random.shuffle(mnist_train[1].numpy())
#
# rng_state = np.random.get_state()
# np.random.shuffle(mnist_val[0].numpy())
# np.random.set_state(rng_state)
# np.random.shuffle(mnist_val[1].numpy())

# envs = [
#   make_environment(mnist_train[0][::2], mnist_train[1][::2], args.train_env_1__color_noise),
#   make_environment(mnist_train[0][1::2], mnist_train[1][1::2], args.train_env_2__color_noise),
#   make_environment(mnist_val[0], mnist_val[1], args.test_env__color_noise),
#   #make_environment(mnist_train[0][:25000:], mnist_train[1][:25000:], args.test_env__color_noise)
# ]
# train_batch_num = envs[0]['images'].shape[0] / args.batch_size
# val_batch_num = envs[2]['images'].shape[0] / args.batch_size
# test_batch_num = envs[2]['images'].shape[0] / args.batch_size
# Define and instantiate the model
if False:
  model = torch.load(args.resume)
else:
  model = maskvgg11(10, cfg)

if use_cuda:
  mlp = model.cuda()
else:
  mlp = model

# Define loss function helpers

criterion = nn.CrossEntropyLoss(reduction='none')
if data_set == 'mnist':
  optimizer = optim.SGD(mlp.parameters(), lr=args.lr, momentum=0.9)
else:
  optimizer = optim.Adam(mlp.parameters(), lr=args.lr)
  # optimizer = optim.SGD(mlp.parameters(), lr=args.lr, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, args.scheduler, gamma=0.25)
pretty_print('step', 'train nll', 'train acc', 'rex penalty', 'irmv1 penalty', 'test acc')
# train_los_pre=None

p1 = args.train_env_1__color_noise
p2 = args.train_env_2__color_noise
p_test = args.test_env__color_noise

env1 = create_env(p1, False, args.batch_size, data_set)
env2 = create_env(p2, False, args.batch_size, data_set)
env_test = create_env(p_test, True, args.batch_size, data_set)
envs = [env1, env2, env_test]



# if args.bn == False and args.cox == True:
#   name = 'Ours+REX'
# elif args.bn == True and args.cox == False:
#   name = 'REX'
# elif args.bn == False and args.cox == False:
#   name = 'No pruning'
wandb.init(project = "Prune_OOD_{}".format(data_set), entity='peilab', name = name)

for epoch in range(args.epochs):

  highest_test_acc = 0.0
  losses = AverageMeter()
  loss_lasso_record=AverageMeter()
  loss_graph_record=AverageMeter()
  
  top1 = AverageMeter()
  top5 = AverageMeter()
  # loss_ce_list=[]
  id_score_collection = []
  ood_score_collection = []
  biased_loss = 0
  unbiased_loss = 0
  biased_data_num = 0
  unbiased_data_num = 0
  train_acc_list = []

  for step in range(len(env1['loader'])):
    n =step
    _mask_list = []
    lasso_list = []
    _mask_before_list = []
    _avg_fea_list = []
    data_loss_increase_p = []
    ood_loss_increase_p = []
    success_p = []

    batch_size = int(args.batch_size)
    mlp.train()

    loss_best_devide_partio = []

    for edx, env in enumerate(envs[:2]):

      samples = next(iter(env["loader"]))
      if len(samples)==2:
        x = samples[0]
        y = samples[1]
        y, domain_label = torch.split(y, 1, dim=1)
        y = y.squeeze().long()
        domain_label = domain_label.squeeze()
      elif len(samples) == 3:
        x = samples[0]
        y = samples[1]
        domain_label = samples[2]
      x = x.cuda().float()
      y = y.cuda().long()
      domain_label = domain_label.cuda()
      logits,env['_mask_list'],env['lasso_list'],env['_mask_before_list'],env['_avg_fea_list']= mlp(x)
      #logits,_mask_list,lasso_list,_mask_before_list,_avg_fea_list= mlp(env['images'][int(n*batch_size):int((n+1)*batch_size)])
      env['nll'] = mean_nll(logits, y)
      env['acc'] = mean_accuracy(logits, y)
      env['penalty'] = penalty(logits, y)
      env['domain_label'] = domain_label
    
    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    train_acc_list.append(train_acc.tolist())

    for l in range(len(env['_mask_list'])):
      _mask_list.append(torch.cat([envs[0]['_mask_list'][l], envs[1]['_mask_list'][l]],dim=0))
      lasso_list.append(torch.cat([envs[0]['lasso_list'][l], envs[1]['lasso_list'][l]],dim=0))
      _mask_before_list.append(torch.cat([envs[0]['_mask_before_list'][l], envs[1]['_mask_before_list'][l]],dim=0))
      _avg_fea_list.append(torch.cat([envs[0]['_avg_fea_list'][l], envs[1]['_avg_fea_list'][l]],dim=0))


    ood_label_total = torch.cat([envs[0]['domain_label'],envs[1]['domain_label']], dim=0)
    id_label_total = 1-ood_label_total
    ood_score_avg = []
    id_score_avg = []
    for i in lasso_list:
      tep = i*ood_label_total
      tep = tep[torch.nonzero(tep)]
      tep = torch.mean(tep)
      ood_score_avg.append(tep.detach().cpu().numpy().tolist())

      tep = i * id_label_total
      tep = tep[torch.nonzero(tep)]
      tep = torch.mean(tep)
      id_score_avg.append(tep.detach().cpu().numpy().tolist())

    id_score_collection.append(id_score_avg)
    ood_score_collection.append(ood_score_avg)

    if use_cuda:
      weight_norm = torch.tensor(0.).cuda()
    else:
      weight_norm = torch.tensor(0.)
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)

    loss1 = envs[0]['nll']
    loss2 = envs[1]['nll']
    loss = 0.0
    loss_lasso=0.0
    loss_each = torch.cat([loss1, loss2],dim=0)
    domain_each = torch.cat([envs[0]["domain_label"], envs[1]["domain_label"]], dim=0)
    loss = args.erm_amount * loss_each.mean()
    #****************************Regularization1: REX Pelnalty)***************************
    irmv1_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
    #penalty_weight = (args.penalty_weight if epoch >= args.penalty_anneal_iters else 1.0)

    if args.mse:
      rex_penalty = (loss1.mean() - loss2.mean()) ** 2
    else:
      rex_penalty = (loss1.mean() - loss2.mean()).abs()

    if args.rex:
      penalty_weight = 1
      loss += penalty_weight * rex_penalty
      if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight
    elif args.irm:
      penalty_weight = 1
      # penalty_weight = args.penalty_weight if epoch >= args.epochs / 5 else 1
      loss += penalty_weight * irmv1_penalty
      # if penalty_weight > 1.0:
      #   # Rescale the entire loss to keep gradients in a reasonable range
      #   loss /= penalty_weight
      loss *=1.5

    elif args.dro:
      q = (0.01 * loss_each).exp()
      q /= q.sum()
      q = q.detach()
      loss = torch.dot(loss_each, q)



    #****************************Regularization2: Complex Pelnalty)***************************
    # loss_ce_list.append(loss_each.data.cpu())

    # train_los_pre = train_los_pre.mean().cuda()

    # loss_avg_ood = torch.sum(loss_each*domain_each)/torch.sum(domain_each)
    # loss_avg_id = torch.sum(loss_each*(1-domain_each))/torch.sum(1-domain_each)
    # loss_devide = (loss_avg_ood+loss_avg_id)/2
    # loss_avg = torch.mean(loss_each).detach()
    # loss_current_partio = loss_devide/loss_avg
    # loss_before_partio = loss_devide/train_los_pre
    # loss_best_devide_partio.append(loss_devide.cpu().detach().numpy().tolist())
    # w1 = (loss_each < args.thre_cls * loss_avg).float(

      # print(w1.sum()/w1.numel())

      # loss_increse_num = (1-w1).sum()
      # data_num = w1.numel()
      # ood_num = domain_each.sum()
      # bingo = (1-w1) * domain_each
      # tep = loss_increse_num / data_num * 100
      # data_loss_increase_p.append(tep.tolist())
      # tep = bingo.sum() / ood_num * 100
      # ood_loss_increase_p.append(tep.tolist())
      # tep = bingo.sum() / loss_increse_num *100
      # success_p.append(tep.tolist())



      #
      # wandb.log({"loss_distance": loss_devide,
      #            "loss_current_partio": loss_current_partio,
      #            "loss_before_partio": loss_before_partio,
      #            "loss_guess_right_ratio": tep.tolist()})

    if args.cox:
      # if train_los_pre != None:
        # w2=((loss_each-args.thre_cls*train_los_pre)/(loss_each.mean()))
      w1 = (loss_each < loss_each.sort()[0][int(args.thre_cls * loss_each.numel())]).float()
      w2=50
      w=w1*w2
      for ilasso in range(len(lasso_list)):
        loss_lasso=loss_lasso+(lasso_list[ilasso]*w).mean()
        # why 'w2' is wrong?
        #iter, layer, avglasso, loss 3 0 tensor(1478.3196, device='cuda:0', grad_fn=<MeanBackward0>) tensor(4.1995, device='cuda:0', grad_fn=<MeanBackward0>) tensor(0.0028, device='cuda:0')
        # mistake 2:w2=((loss_each - args.thre_cls*train_los_pre)/(loss_each)) the denominator lost .mean()

      loss += args.lambda_lasso*loss_lasso
    #****************************Regularization3: Sparse Pelnalty)*****************
    if args.bn:
      w2 = 50
      for ilasso in range(len(lasso_list)):
          loss_lasso=loss_lasso + (lasso_list[ilasso]*w2).mean()
      loss += args.lambda_lasso*loss_lasso

    loss += args.l2_regularizer_weight * weight_norm
    #****************************************************************************
    unbiased_loss += (loss_each*ood_label_total).sum().detach()
    unbiased_data_num += ood_label_total.sum()
    biased_loss += (loss_each*(1-ood_label_total)).sum().detach()
    biased_data_num += (1-ood_label_total).sum()

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(mlp.parameters(), max_norm=1, norm_type=2)
    #updateSaliencyBlock(0.00001, mlp)
    optimizer.step()
  scheduler.step()
  # train_los_pre=torch.cat(loss_ce_list,dim=0)

  info_dict = {
    "epoch": epoch,
    "lr": optimizer.param_groups[0]['lr'],
    "loss increase possibility": np.average(data_loss_increase_p),
    "ood loss increase possibility": np.average(ood_loss_increase_p),
    "loss_distance": np.average(loss_best_devide_partio),
    "effective ratio": np.average(success_p),
    "avg_biased_loss": biased_loss/biased_data_num,
    "avg_unbiased_loss": unbiased_loss/unbiased_data_num,
    "train_acc": np.mean(train_acc_list)
  }

  ood_score_collection = np.array(ood_score_collection)
  id_score_collection = np.array(id_score_collection)
  ood_score_collection = np.mean(ood_score_collection, axis=0)
  id_score_collection = np.mean(id_score_collection, axis=0)

  for layer in range(len(id_score_collection)):
    info_dict["id score of layer {}".format(layer)] = id_score_collection[layer]
    info_dict["ood score of layer {}".format(layer)] = ood_score_collection[layer]
  info_dict["general id score"] = id_score_collection.mean()
  info_dict["general ood score"] = ood_score_collection.mean()
  wandb.log(info_dict)
  if epoch % args.eval_interval == 0:
    mlp.eval()
    with torch.no_grad():
      # x, y = next(iter(envs[2]["loader"]))
      # x = x.cuda()
      # y = y.cuda().long()
      # y, domain_label = torch.split(y, 1,dim=1)
      # logits, _mask_list, lasso_list, _mask_before_list, _avg_fea_list = mlp(x)
      # envs[2]['nll'] =  mean_nll(logits,y)
      # envs[2]['acc'] =  mean_accuracy(logits,y)
      # test_acc = envs[2]['acc']*args.batch_size / envs[2]["loader"].batch_size

      samples = next(iter(envs[2]["loader_id"]))
      if len(samples) == 2:
        x = samples[0]
        y = samples[1]
        y, domain_label = torch.split(y, 1, dim=1)
        y = y.squeeze().long()
        domain_label = domain_label.squeeze()
      elif len(samples) == 3:
        x = samples[0]
        y = samples[1]
        domain_label = samples[2]
      x = x.cuda().float()
      y = y.cuda().long()
      domain_label = domain_label.cuda()
      logits, _mask_list, lasso_list, _mask_before_list, _avg_fea_list = mlp(x)
      envs[2]['nll_id'] = mean_nll(logits, y)
      envs[2]['acc_id'] = mean_accuracy(logits, y)
      id_acc = envs[2]['acc_id']*args.batch_size / envs[2]["loader"].batch_size

      samples = next(iter(envs[2]["loader_ood"]))
      if len(samples) == 2:
        x = samples[0]
        y = samples[1]
        y, domain_label = torch.split(y, 1, dim=1)
        y = y.squeeze().long()
        domain_label = domain_label.squeeze()
      elif len(samples) == 3:
        x = samples[0]
        y = samples[1]
        domain_label = samples[2]
      x = x.cuda().float()
      y = y.cuda().long()
      domain_label = domain_label.cuda()
      logits, _mask_list, lasso_list, _mask_before_list, _avg_fea_list = mlp(x)
      envs[2]['nll_ood'] = mean_nll(logits, y)
      envs[2]['acc_ood'] = mean_accuracy(logits, y)
      class_acc = class_accuracy(logits, y)
      envs[2]['class_acc'] = class_acc
      ood_acc = envs[2]['acc_ood']*args.batch_size / envs[2]["loader"].batch_size
    train_acc_scalar = train_acc.detach().cpu().numpy()
    # test_acc_scalar = test_acc.detach().cpu().numpy()
    # if (train_acc_scalar >= test_acc_scalar) and (test_acc_scalar > highest_test_acc):
    #   highest_test_acc = test_acc_scalar
    # all_test_accs[epoch, step] = test_acc.detach().cpu().numpy()
    
    if args.print_eval_intervals:
      # pretty_print(
      #   np.int32(epoch),
      #   train_nll.detach().cpu().numpy(),
      #   train_acc.detach().cpu().numpy(),
      #   test_acc.detach().cpu().numpy()
      # )
      info_dict = {
        "epoch": epoch,
        # "train_loss": train_nll,
        # "train_acc": train_acc,
        # "test_acc": test_acc,
        "id_acc": id_acc,
        "ood_acc": ood_acc
      }
      for i in range(len(envs[2]['class_acc'])):
        info_dict['class_{}_acc'.format(i)] = envs[2]['class_acc'][i]
      wandb.log(info_dict)

      logging.info("epoch: [{}]\t"
           "Train Loss {train_nll:.3f}\t"
           "Train Acc@1 {train_acc:.3f}\t"
           "test_acc Acc@1 {test_acc:.3f}\t"
           .format(epoch, train_nll = train_nll, train_acc=train_acc, test_acc=ood_acc))
      
      if args.plot or args.save_numpy_log:
        all_train_nlls[epoch, step] = train_nll.detach().cpu().numpy()
        all_train_accs[epoch, step] = train_acc.detach().cpu().numpy()
        all_rex_penalties[epoch, step] = rex_penalty.detach().cpu().numpy()
        all_irmv1_penalties[epoch, step] = irmv1_penalty.detach().cpu().numpy()
sparse_model = mlp
torch.save(sparse_model,os.path.join(root,args.savepath))

# x, y = next(iter(envs[1]["loader"]))
# pruned_model = prune_while_training(sparse_model, num_classes=num_classes, data = x.cpu())
# torch.save(pruned_model,os.path.join(root,args.pruned_savepath))



'''
freeze_mask(mlp)
ft_optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
ft_scheduler = lr_scheduler.MultiStepLR(optimizer, [60,100], gamma=0.1,last_epoch=-1)
for epoch in range(args.fine_tune_epochs):
  highest_test_acc = 0.0  
  for step in range(int(train_batch_num)):
    n =step
    #n = i % train_batch_num                       
    batch_size = int(args.batch_size)
    mlp.train()
    for edx, env in enumerate(envs[:2]):  
      
      logits,_mask_list,lasso_list,_mask_before_list,_avg_fea_list= mlp(env['images'][int(n*batch_size):int((n+1)*batch_size)])
      env['nll'] = mean_nll(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
      env['acc'] = mean_accuracy(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
      env['penalty'] = penalty(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
    
    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    irmv1_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

    loss1 = envs[0]['nll']
    loss2 = envs[1]['nll']
    loss = args.erm_amount * (loss1 + loss2).mean()

    penalty_weight = (args.penalty_weight if epoch >= args.penalty_anneal_iters else 1.0)    
    if args.mse:
      rex_penalty = (loss1.mean() - loss2.mean()) ** 2
    else:
      rex_penalty = (loss1.mean() - loss2.mean()).abs()
    if args.rex:
      loss += penalty_weight * rex_penalty
    else:
      loss += penalty_weight * irmv1_penalty
    if penalty_weight > 1.0:
      # Rescale the entire loss to keep gradients in a reasonable range
      loss /= penalty_weight
    
    ft_optimizer.zero_grad()
    loss.backward()
    ft_optimizer.step()
    ft_scheduler.step()

  if epoch % args.eval_interval == 0:
    mlp.eval()
    with torch.no_grad():
      logits,_mask_list,lasso_list,_mask_before_list,_avg_fea_list= mlp(envs[2]['images'])
      envs[2]['nll'] =  mean_nll(logits,envs[2]['labels'])
      envs[2]['acc'] =  mean_accuracy(logits,envs[2]['labels'])                     
      test_acc = envs[2]['acc'] / val_batch_num
    train_acc_scalar = train_acc.detach().cpu().numpy()
    test_acc_scalar = test_acc.detach().cpu().numpy()
    if (train_acc_scalar >= test_acc_scalar) and (test_acc_scalar > highest_test_acc):
      highest_test_acc = test_acc_scalar
  
    
    if args.print_eval_intervals:
      pretty_print(
        np.int32(epoch),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )
      logging.info("epoch: [{}]\t"
           "Train Loss {train_nll:.3f}\t"
           "Train Acc@1 {train_acc:.3f}\t"
           "test_acc Acc@1 {test_acc:.3f}\t"
           .format(epoch, train_nll = train_nll, train_acc=train_acc, test_acc=test_acc))
'''
# print('highest test acc this run:', highest_test_acc)
# logging.info('highest test acc this run: {}'.format(highest_test_acc))

# final_train_accs.append(train_acc.detach().cpu().numpy())
# final_test_accs.append(test_acc.detach().cpu().numpy())
# highest_test_accs.append(highest_test_acc)
# print('Final train acc (mean/std across on epoch {} so far):')
# print(np.mean(final_train_accs), np.std(final_train_accs))
# logging.info('Final train acc (mean/std across restarts so far): {} / {}'.format(epoch,np.mean(final_train_accs), np.std(final_train_accs)))
#
# print('Final test acc (mean/std across on epoch {} so far):')
# print(np.mean(final_test_accs), np.std(final_test_accs))
# logging.info('Final test acc (mean/std across restarts so far): {} / {}'.format(np.mean(final_test_accs), np.std(final_test_accs)))

# print('Highest test acc (mean/std across on epoch {} so far):')
# print(np.mean(highest_test_accs), np.std(highest_test_accs))
# logging.info('Highest test acc (mean/std across restarts so far): {} / {}'.format(np.mean(highest_test_accs), np.std(highest_test_accs)))
# if args.plot:
#   plot_x = np.linspace(0, args.steps, args.steps)
#   from pylab import *
#
#   figure()
#   xlabel('epoch')
#   ylabel('loss')
#   title('train/test accuracy')
#   plot(plot_x, all_train_accs.mean(0), ls="dotted", label='train_acc')
#   plot(plot_x, all_test_accs.mean(0), label='test_acc')
#   plot(plot_x, all_grayscale_test_accs.mean(0), ls="--", label='grayscale_test_acc')
#   legend(prop={'size': 11}, loc="upper right")
#   savefig('train_acc__test_acc.pdf')
#
#   figure()
#   title('train nll / penalty ')
#   plot(plot_x, all_train_nlls.mean(0), ls="dotted", label='train_nll')
#   plot(plot_x, all_irmv1_penalties.mean(0), ls="--", label='irmv1_penalty')
#   plot(plot_x, all_rex_penalties.mean(0), label='rex_penalty')
#   yscale('log')
#   legend(prop={'size': 11}, loc="upper right")
#   savefig('train_nll__penalty.pdf')
#
# if args.save_numpy_log:
#   import os
#   directory = "np_arrays_paper"
#   if not os.path.exists(directory):
#     os.makedirs(directory)
#
#   outfile = "all_train_nlls"
#   np.save(directory + "/" + outfile, all_train_nlls)
#
#   outfile = "all_irmv1_penalties"
#   np.save(directory + "/" + outfile, all_irmv1_penalties)
#
#   outfile = "all_rex_penalties"
#   np.save(directory + "/" + outfile, all_rex_penalties)
#
#   outfile = "all_train_accs"
#   np.save(directory + "/" + outfile, all_train_accs)
#
#   outfile = "all_test_accs"
#   np.save(directory + "/" + outfile, all_test_accs)
#
#   outfile = "all_grayscale_test_accs"
#   np.save(directory + "/" + outfile, all_grayscale_test_accs)
#
