import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
import torch.utils.data as Data
import torchvision 
from manivgg import maskvgg11
import  cv2
import os
import logging
import math
from logging import FileHandler
from logging import StreamHandler
import torch.optim.lr_scheduler as lr_scheduler
from common import  *
from prun import *
from utils import *
import resnet
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

use_cuda = torch.cuda.is_available()


parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--squeeze_rate', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--scheduler', type=list, default=[271,420])
parser.add_argument('--epochs', type=int, default=521)
parser.add_argument('--fine_tune_epochs', type=int, default=121)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--fine_tune', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--train_set_size', type=int, default=50000)
parser.add_argument('--eval_interval', type=int, default=10)
parser.add_argument('--print_eval_intervals', type=str2bool, default=True)
parser.add_argument('--polar', type=bool, default=True)
parser.add_argument('--train_env_1__color_noise', type=float, default=0.8)
parser.add_argument('--train_env_2__color_noise', type=float, default=0.9)
#parser.add_argument('--val_env__color_noise', type=float, default=0.1)
parser.add_argument('--test_env__color_noise', type=float, default=0.2)

parser.add_argument('--erm_amount', type=float, default=1.0)

parser.add_argument('--early_loss_mean', type=str2bool, default=True)

parser.add_argument('--rex', type=str2bool, default=True)
parser.add_argument('--mse', type=str2bool, default=False)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--plot', type=str2bool, default=False)
parser.add_argument('--save_numpy_log', type=str2bool, default=False)
parser.add_argument('--thre_init', type=float, default=-10000.0)
parser.add_argument('--thre_cls', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=1.0)

parser.add_argument('--lambda_lasso', type=float, default=0.1,help='group lasso loss weight')
parser.add_argument('--target_remain_rate', type=float, default=0.65)
parser.add_argument('--logpath', type=str, default='sparse_2.txt')
parser.add_argument('--savepath', type=str, default='sparse_2.pth.tar')
parser.add_argument('--prunedpath', type=str, default='pruned_2.pth.tar')
parser.add_argument('--resume', type=str, default="/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/checkpoint/full_2.pth.tar")
args = parser.parse_args()
root='/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/checkpoint'

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
def updateBN(lbd,mlp):
    
  sparsity = lbd
  bn_modules = list(filter(lambda m: (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)), mlp.named_children()))
  #bn_modules = list(map(lambda m: m, bn_modules))  # remove module name
  for m in bn_modules:
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
          m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))
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


def penalty(logits, y):
  if use_cuda:
    scale = torch.tensor(1.).cuda().requires_grad_()
  else:
    scale = torch.tensor(1.).requires_grad_()
  
  loss =mean_nll(logits * scale, y.squeeze())
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
def train(mlp, lambda_lasso, lambda_graph, savepath, prunedpath):
#none: no avg for losses of each batch. no none:compute avg for losses of baches
  
  optimizer = optim.SGD(mlp.parameters(), lr=args.lr, momentum=0.9)
  scheduler = lr_scheduler.MultiStepLR(optimizer, args.scheduler, gamma=0.1,last_epoch=-1)
  pretty_print('step', 'train nll', 'train acc', 'rex penalty', 'irmv1 penalty', 'test acc')
  train_los_pre=None
  
  for epoch in range(args.epochs):
  
    highest_test_acc = 0.0
    losses = AverageMeter()
    loss_lasso_record=AverageMeter()
    loss_graph_record=AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_ce_list=[]  
  
    
    for step in range(int(train_batch_num)):
      n =step                  
      batch_size = int(args.batch_size)
      #n = i % train_batch_num                       
      batch_size = int(args.batch_size)
      mlp.train()
  
      for edx, env in enumerate(envs[:2]):  
        #logits,env['_mask_list'],env['lasso_list'],env['_mask_before_list'],env['_avg_fea_list']= mlp(env['images'][int(n*batch_size):int((n+1)*batch_size)])
        logits,_mask_list,lasso_list,_mask_before_list,_avg_fea_list= mlp(env['images'][int(n*batch_size):int((n+1)*batch_size)])
        env['nll'] = mean_nll(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
        env['acc'] = mean_accuracy(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
        env['penalty'] = penalty(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
      
      train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
      train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
   
      
      if use_cuda:
        weight_norm = torch.tensor(0.).cuda()
      else:
        weight_norm = torch.tensor(0.)
      for w in mlp.parameters():
        weight_norm += w.norm().pow(2)
      
      loss1 = envs[0]['nll']#loss1 torch.Size([5000])
      loss2 = envs[1]['nll']
      loss = 0.0
      loss_lasso=0.0  
      loss_each = loss1 + loss2
    
      loss = args.erm_amount * loss_each.mean()
      
      
      #****************************Regularization1: Our Designed Prun Method For OOD)***************************
      '''
      loss_ce_list.append(loss_each.data.cpu())
      if train_los_pre != None:
        train_los_pre=train_los_pre.mean().cuda()
        w1=(loss_each < args.thre_cls*train_los_pre).float() 
        w2=((args.thre_cls*train_los_pre-loss_each)/(args.thre_cls*train_los_pre))#w2,loss_each torch.Size([10000]) torch.Size([10000])        
        w=w1*w2                 
        for ilasso in range(len(lasso_list)):
          loss_lasso=loss_lasso+(lasso_list[ilasso]*w).mean()     
  
        loss=loss_each.mean() + lambda_lasso*loss_lasso
    
    #****************************Regularization2: Our Designed Prun Method For OOD)***************************      
      
      irmv1_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
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
      '''
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()
    #train_los_pre=torch.cat(loss_ce_list,dim=0)
    
  
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
      all_test_accs[epoch, step] = test_acc.detach().cpu().numpy()
      
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
        
        if args.plot or args.save_numpy_log:
          all_train_nlls[epoch, step] = train_nll.detach().cpu().numpy()
          all_train_accs[epoch, step] = train_acc.detach().cpu().numpy()
          all_rex_penalties[epoch, step] = rex_penalty.detach().cpu().numpy()
          all_irmv1_penalties[epoch, step] = irmv1_penalty.detach().cpu().numpy()
  torch.save(mlp,os.path.join(root,savepath))
  pruned_model = prune_while_training(mlp, num_classes=num_classes, data = envs[2]['images'].cpu())
  torch.save(pruned_model,os.path.join(root,prunedpath)) 

color_dict = {'0': [255,0,0], '1': [255,255,0], '2': [0,255,0], '3': [0,100,0], '4': [0,0,255], '5': [255, 0,255],'6': [0,0,128], '7': [220,220,220], '8': [255,255,255], '9': [0,255,255]}
num_classes = 10
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

envs = [
  make_environment(mnist_train[0][::2], mnist_train[1][::2], args.train_env_1__color_noise),
  make_environment(mnist_train[0][1::2], mnist_train[1][1::2], args.train_env_2__color_noise),
  make_environment(mnist_val[0], mnist_val[1], args.test_env__color_noise),
  #make_environment(mnist_train[0][:25000:], mnist_train[1][:25000:], args.test_env__color_noise)
]  
train_batch_num = envs[0]['images'].shape[0] / args.batch_size 
val_batch_num = envs[2]['images'].shape[0] / args.batch_size 
test_batch_num = envs[2]['images'].shape[0] / args.batch_size 



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
# Define and instantiate the model
if args.resume != None:
  #model=resnet.resnet20(10,args).cuda()
  model = torch.load(args.resume).cuda()
  #model2 = torch.load(args.resume).cuda()
else:
  model = maskvgg11(10).cuda()

criterion = nn.CrossEntropyLoss(reduction='none')
# Define loss function helpers
train(model, 0, 0, savepath=args.savepath, prunedpath=args.prunedpath)
#train(model1, args.lambda_lasso, savepath=os.path.join('cop_'+args.savepath), prunedpath=os.path.join('cop_'+args.prunedpath))
#train(model2, 0,savepath=os.path.join('rex_'+args.savepath), prunedpath=os.path.join('rex_'+args.prunedpath))


if args.plot:
  plot_x = np.linspace(0, args.steps, args.steps)
  from pylab import *

  figure()
  xlabel('epoch')
  ylabel('loss')
  title('train/test accuracy')
  plot(plot_x, all_train_accs.mean(0), ls="dotted", label='train_acc')
  plot(plot_x, all_test_accs.mean(0), label='test_acc')
  plot(plot_x, all_grayscale_test_accs.mean(0), ls="--", label='grayscale_test_acc')
  legend(prop={'size': 11}, loc="upper right")
  savefig('train_acc__test_acc.pdf')

  figure()
  title('train nll / penalty ')
  plot(plot_x, all_train_nlls.mean(0), ls="dotted", label='train_nll')
  plot(plot_x, all_irmv1_penalties.mean(0), ls="--", label='irmv1_penalty')
  plot(plot_x, all_rex_penalties.mean(0), label='rex_penalty')
  yscale('log')
  legend(prop={'size': 11}, loc="upper right")
  savefig('train_nll__penalty.pdf')

if args.save_numpy_log:
  import os
  directory = "np_arrays_paper"
  if not os.path.exists(directory):
    os.makedirs(directory)

  outfile = "all_train_nlls"
  np.save(directory + "/" + outfile, all_train_nlls)

  outfile = "all_irmv1_penalties"
  np.save(directory + "/" + outfile, all_irmv1_penalties)

  outfile = "all_rex_penalties"
  np.save(directory + "/" + outfile, all_rex_penalties)

  outfile = "all_train_accs"
  np.save(directory + "/" + outfile, all_train_accs)

  outfile = "all_test_accs"
  np.save(directory + "/" + outfile, all_test_accs)

  outfile = "all_grayscale_test_accs"
  np.save(directory + "/" + outfile, all_grayscale_test_accs)
  
  '''
  class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if args.grayscale_model:
        lin1 = nn.Linear(14 * 14, args.hidden_dim)
      else:
        lin1 = nn.Linear(2 * 14 * 14, args.hidden_dim)
      lin2 = nn.Linear(args.hidden_dim, args.hidden_dim)
      lin3 = nn.Linear(args.hidden_dim, 1)
      for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    def forward(self, input):
      if args.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.view(input.shape[0], 2 * 14 * 14)
      out = self._main(out)
      return out
      
  train_env = make_environment(mnist_train[0][::1], mnist_train[1][::1], 1.0)
  val_env = make_environment(mnist_val[0][::1], mnist_val[1][::1], 1.0)
  test_env = make_environment(mnist_test[0][::1], mnist_test[1][::1], 0.0)

  train_data.data = train_env['images'].reshape(train_env['images'].shape[0],3*28,28)
  train_data.targets = train_env['labels']
  val_data.data = val_env['images'].reshape(val_env['images'].shape[0],3*28,28)
  val_data.targets = val_env['labels']
  test_data.data = test_env['images'].reshape(test_env['images'].shape[0],3*28,28)
  test_data.targets = test_env['labels']
  
  train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
  val_loader = Data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
  test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
  
  train_batch_num = len(train_loader)
  val_batch_num = len(val_loader)
  test_batch_num = len(test_loader)
  
  pruned_model.cuda()
  with torch.no_grad():
    logits = pruned_model(envs[2]['images'])
    envs[2]['nll'] =  mean_nll(logits,envs[2]['labels'])
    envs[2]['acc'] =  mean_accuracy(logits,envs[2]['labels'])                     
    pruned_acc = envs[2]['acc']
  print('**********prund_acc: ',pruned_acc)
  
  
  '''