import argparse
import numpy as np
import torch
from dataset import MNIST_colored
from dataset import prepare_ood_colored_mnist
from dataset import manual_seed
import torch.nn.functional as F
from torch import nn, optim, autograd
import torch.utils.data as Data
import torchvision
import  wandb
from maskvgg import maskvgg11
import  cv2
import logging
import math
from logging import FileHandler
from logging import StreamHandler
import torch.optim.lr_scheduler as lr_scheduler
from common import  *
from prun import *
import re
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--scheduler', type=list, default=[60])
parser.add_argument('--epochs', type=int, default=241)
parser.add_argument('--fine_tune_epochs', type=int, default=101)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--train_set_size', type=int, default=50000)
parser.add_argument('--eval_interval', type=int, default=10)
parser.add_argument('--print_eval_intervals', type=str2bool, default=True)

parser.add_argument('--train_env_1__color_noise', type=float, default=0.8)
parser.add_argument('--train_env_2__color_noise', type=float, default=0.9)
#parser.add_argument('--val_env__color_noise', type=float, default=0.1)
parser.add_argument('--test_env__color_noise', type=float, default=0.2)

parser.add_argument('--erm_amount', type=float, default=1.0)

parser.add_argument('--early_loss_mean', type=str2bool, default=True)

parser.add_argument('--rex', type=str2bool, default=True)
parser.add_argument('--cox', type=str2bool, default=True)
parser.add_argument('--sim', type=str2bool, default=True)
parser.add_argument('--bn', type=str2bool, default=False)
parser.add_argument('--mse', type=str2bool, default=False)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--plot', type=str2bool, default=False)
parser.add_argument('--save_numpy_log', type=str2bool, default=False)
parser.add_argument('--thre_init', type=float, default=-10000.0)
parser.add_argument('--thre_cls', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--lambda_lasso', type=float, default=0.01,help='group lasso loss weight')
parser.add_argument('--logpath', type=str, default='finetune.txt')
parser.add_argument('--fintune_savepath', type=str, default='finetune.pth.tar')
#parser.add_argument('--resume', type=str, default="/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/checkpoint_2/nslim_pruned.pth.tar")
args = parser.parse_args()

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
num_batches = (args.train_set_size // 2) // args.batch_size
if args.gpu != None: 
    torch.cuda.set_device(args.gpu)


final_train_accs = []
final_test_accs = []
highest_test_accs = []

  
  
color_dict = {'0': [255,0,0], '1': [255,255,0], '2': [0,255,0], '3': [0,100,0], '4': [0,0,255], '5': [255, 0,255],'6': [0,0,128], '7': [220,220,220], '8': [255,255,255], '9': [0,255,255]}
num_classes = 10


def create_env(p, val=False, batch_size=5000):
  # if os.path.exists('Mixed_Mnist_train_{}.pt'.format(p)):
  #   pass
  # else:
  mixed_train, mixed_test = prepare_ood_colored_mnist('mnist', p)
  if val:
    loader = torch.utils.data.DataLoader(mixed_test, len(mixed_test), shuffle=True)
  else:
    loader = torch.utils.data.DataLoader(mixed_train, batch_size, shuffle=True)
  return {
    "loader": loader
  }



train_data = torchvision.datasets.MNIST(root='/data/wyc', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
val_data = torchvision.datasets.MNIST(root='/data/wyc',
                                      transform=torchvision.transforms.ToTensor(),
                                      train=False)
test_data = torchvision.datasets.MNIST(root='/data/wyc',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)
mnist_train = (train_data.data[:50000], train_data.targets[:50000])
mnist_val = (train_data.data[40000:60000], train_data.targets[40000:60000])
mnist_test = (test_data.data[:10000], test_data.targets[:10000])
rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

rng_state = np.random.get_state()
np.random.shuffle(mnist_val[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_val[1].numpy())
p1 = args.train_env_1__color_noise
p2 = args.train_env_2__color_noise
p_test = args.test_env__color_noise

env1 = create_env(p1, False, args.batch_size)
env2 = create_env(p2, False, args.batch_size)
env_test = create_env(p_test, True, args.batch_size)
envs = [env1, env2, env_test]
# if args.bn == False and args.cox == True:
#   name = 'Ours+REX'
# elif args.bn == True and args.cox == False:
#   name = 'Nslim+REX'


def model_reset(model: nn.Module):
    print('config:',model.config())
    pruned_model = vgg11(10,model.config())
    '''
    new_state_dict = {}
    for n,p in model.named_parameters():
        if n in pruned_model.state_dict():
            new_state_dict[n] = p
    pruned_model.load_state_dict(new_state_dict,strict=False)
    '''
    return pruned_model  

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
# Define loss function helpers

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
  top1 = top1.mul_(100.0 / args.batch_size)

  return top1


def penalty(logits, y):
  if use_cuda:
    scale = torch.tensor(1.).cuda().requires_grad_()
  else:
    scale = torch.tensor(1.).requires_grad_()
  loss = mean_nll(logits * scale, y.squeeze())
  grad = autograd.grad(loss.mean(), [scale], create_graph=True)[0]
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
#
# def make_environment(images, labels, e):
#   def torch_bernoulli(p, size):
#     return (torch.rand(size) < p).float()
#   def torch_xor(a, b):
#     return (a-b).abs() # Assumes both inputs are either 0 or 1
#
#   images = images.reshape((-1, 28, 28))
#   #images = np.array([cv2.resize(images[i].detach().cpu().numpy().astype(np.uint8),(48,48)) for i in range(images.shape[0])])
#   #images = torch.tensor(images.reshape(-1,48,48)).cuda()
#
#   images = torch.stack([images, images, images], dim=3)
#
#   # Assign a color based on the label; flip the color with probability e
#   for img in range(len(images)):
#       args = torch_bernoulli(e, len(labels))
#       label = labels[img]
#
#       if args[img] > 0:
#           color = color_dict[str(int(np.array(label)))]
#       else:
#           color = color_dict[str(np.array(torch.randint(10,[1,]))[0])]
#       for rgb in range(3):
#           c_color = color[rgb]
#
#           images[img, :, :, rgb] *= c_color
#
#   images =  images.reshape(-1,3,28,28)[:, :, ::2, ::2]
#
#   all_image = (images.float() / 255.).cuda()
#   all_label = labels[:, None].cuda()
#   return {
#     'images': (images.float() / 255.).cuda(),
#     'labels': labels[:, None].cuda()
#   }
def train(mlp, lambda_lasso,finetunepath, name):
  wandb.init(project="Prune_OOD", entity='peilab', name=name)
  optimizer = optim.SGD(mlp.parameters(), lr=args.lr, momentum=0.9)
  scheduler = lr_scheduler.MultiStepLR(optimizer, args.scheduler, gamma=0.1,last_epoch=-1)
  pretty_print('step', 'train nll', 'train acc', 'rex penalty', 'irmv1 penalty', 'test acc')
  train_los_pre=None
  for epoch in range(args.fine_tune_epochs):
  
    highest_test_acc = 0.0
    losses = AverageMeter()
    loss_lasso_record=AverageMeter()
    loss_graph_record=AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_ce_list=[]  
  
  
    for step in range(int(train_batch_num)):
      n =step
      _mask_list = []
      lasso_list = []
      _mask_before_list = []
      _avg_fea_list = []
      #n = i % train_batch_num                       
      batch_size = int(args.batch_size)
      mlp.train()
      for edx, env in enumerate(envs[:2]):
        x, y = next(iter(env["loader"]))
        x = x.cuda()
        y = y.cuda()
        y, domain_label = torch.split(y, 1, dim=1)
        y = y.squeeze().long()
        domain_label = domain_label.squeeze()
        logits,env['_mask_list'],env['lasso_list'],env['_mask_before_list'],env['_avg_fea_list']= mlp(x)
        env['nll'] = mean_nll(logits, y)
        env['acc'] = mean_accuracy(logits, y)
        env['penalty'] = penalty(logits, y)
  
      train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
      train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
      for l in range(len(env['_mask_list'])):
        _mask_list.append(torch.cat([envs[0]['_mask_list'][l], envs[1]['_mask_list'][l]],dim=0))
        lasso_list.append(torch.cat([envs[0]['lasso_list'][l], envs[1]['lasso_list'][l]],dim=0))
        _mask_before_list.append(torch.cat([envs[0]['_mask_before_list'][l], envs[1]['_mask_before_list'][l]],dim=0))
        _avg_fea_list.append(torch.cat([envs[0]['_avg_fea_list'][l], envs[1]['_avg_fea_list'][l]],dim=0))
      loss1 = envs[0]['nll']
      loss2 = envs[1]['nll']
      loss = 0.0      
      loss_each = loss1 + loss2
      loss += args.erm_amount * loss_each.mean()
      
      #****************************Regularization1: REX Method For OOD)***************************    
      
      penalty_weight = (args.penalty_weight if epoch >= args.penalty_anneal_iters else 1.0)
      irmv1_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
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
      '''
      #****************************Regularization2: Complex Method For OOD)***************************   
      loss_lasso=0.0
      loss_ce_list.append(loss_each.data.cpu())
      if lambda_lasso != 0:
        
        if train_los_pre != None:
          train_los_pre=train_los_pre.mean().cuda()
          w1=(loss_each < args.thre_cls*train_los_pre).float() 
          w2=(args.thre_cls*train_los_pre-loss_each)/(args.thre_cls*train_los_pre)
          w=w1*w2                 
          for ilasso in range(len(lasso_list)):
            loss_lasso=loss_lasso+(lasso_list[ilasso]*w).mean()     
          loss=loss_each.mean() + lambda_lasso*loss_lasso
  
      '''
        
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()
  
    #train_los_pre=torch.cat(loss_ce_list,dim=0)
    if epoch % args.eval_interval == 0:
      x, y = next(iter(envs[2]["loader"]))
      x = x.cuda()
      y = y.cuda().long()
      y, domain_label = torch.split(y, 1, dim=1)
      mlp.eval()
      with torch.no_grad():
        logits,_mask_list,lasso_list,_mask_before_list,_avg_fea_list= mlp(x)
        envs[2]['nll'] =  mean_nll(logits,y)
        envs[2]['acc'] =  mean_accuracy(logits,y)
        test_acc = envs[2]['acc']/val_batch_num*args.batch_size / envs[2]["loader"].batch_size
      train_acc_scalar = train_acc.detach().cpu().numpy()
      test_acc_scalar = test_acc.detach().cpu().numpy()
      if (train_acc_scalar >= test_acc_scalar) and (test_acc_scalar > highest_test_acc):
        highest_test_acc = test_acc_scalar
      
      
      if args.print_eval_intervals:
        # pretty_print(
        #   np.int32(epoch),
        #   train_nll.detach().cpu().numpy(),
        #   train_acc.detach().cpu().numpy(),
        #
        #   test_acc.detach().cpu().numpy()
        # )
        logging.info("epoch: [{}]\t"
             "Train Loss {train_nll:.3f}\t"
             "Train Acc@1 {train_acc:.3f}\t"
              #"rex_penalty@1 {rex_penalty:.3f}\t"
             #"irmv1_penalty@1 {irmv1_penalty:.3f}\t"
             "test_acc Acc@1 {test_acc:.3f}\t"
             .format(epoch, train_nll = train_nll, train_acc=train_acc, test_acc=test_acc))
        info_dict = {
          "epoch": epoch,
          "train_loss": train_nll,
          "train_acc": train_acc,
          "test_acc": test_acc
        }
        wandb.log(info_dict)
  
  torch.save(mlp,os.path.join(root,finetunepath))
  wandb.finish()
train_data = torchvision.datasets.MNIST(root='/data/wyc', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
val_data = torchvision.datasets.MNIST(root='/data/wyc',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)
test_data = torchvision.datasets.MNIST(root='/data/wyc',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)

                                 
mnist_train = (train_data.data[:50000], train_data.targets[:50000])
mnist_val = (train_data.data[40000:60000], train_data.targets[40000:60000])
mnist_test = (test_data.data[:10000], test_data.targets[:10000])
rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

rng_state = np.random.get_state()
np.random.shuffle(mnist_val[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_val[1].numpy())

# envs = [
#   make_environment(mnist_train[0][::2], mnist_train[1][::2], args.train_env_1__color_noise),
#   make_environment(mnist_train[0][1::2], mnist_train[1][1::2], args.train_env_2__color_noise),
#   make_environment(mnist_val[0], mnist_val[1], args.test_env__color_noise),
#   #make_environment(mnist_train[0][:25000:], mnist_train[1][:25000:], args.test_env__color_noise)
# ]
# train_batch_num = envs[0]['images'].shape[0] / args.batch_size
# val_batch_num = envs[2]['images'].shape[0] / args.batch_size
# test_batch_num = envs[2]['images'].shape[0] / args.batch_size

train_batch_num = len(envs[0]['loader'])
val_batch_num = len(envs[2]['loader'])
test_batch_num = len(envs[2]['loader'])




# Define and instantiate the model
if True:
    #mlp = torch.load(args.resume).cuda()
    mlp1 = torch.load("./check_point/cox_pruned.pth.tar").cuda()
    mlp2 = torch.load("./check_point/normal_pruned.pth.tar").cuda()
else:
    mlp1 = vgg11(10,cfg=[22, 'M', 56, 'M', 113, 95, 'M', 193, 208, 'M', 225, 235]).cuda()
    mlp2 = vgg11(10,cfg=[23, 'M', 55, 'M', 110, 97, 'M', 191, 212, 'M', 213, 257]).cuda()
criterion = nn.CrossEntropyLoss(reduction='none')
train(mlp1, args.lambda_lasso, finetunepath=os.path.join('ours_'+args.fintune_savepath), name='Ours')
train(mlp2, 0, finetunepath=os.path.join('normal'+args.fintune_savepath), name='Nslim')

