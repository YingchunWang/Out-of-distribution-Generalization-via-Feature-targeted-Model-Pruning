import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
import torch.utils.data as Data
import torchvision 
from maskvgg import maskvgg11
import  cv2
import logging
import math
from logging import FileHandler
from logging import StreamHandler
import torch.optim.lr_scheduler as lr_scheduler
from common import  *
from prun import *
import resnet
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

use_cuda = torch.cuda.is_available()
logger_file = '/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/test.txt'
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

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--thre_cls', type=float, default=0.5)
parser.add_argument('--scheduler', type=list, default=[80,120])
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--fine_tune_epochs', type=int, default=101)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', type=str2bool, default=False)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--train_set_size', type=int, default=50000)
parser.add_argument('--eval_interval', type=int, default=10)
parser.add_argument('--print_eval_intervals', type=str2bool, default=True)
parser.add_argument('--polar', type=bool, default=True)
parser.add_argument('--train_env_1__color_noise', type=float, default=0.8)
parser.add_argument('--train_env_2__color_noise', type=float, default=0.9)
#parser.add_argument('--val_env__color_noise', type=float, default=0.1)
parser.add_argument('--test_env__color_noise', type=float, default=0.2)
parser.add_argument('--squeeze_rate', type=int, default=2)
parser.add_argument('--thre_freq', type=int, default=1)
parser.add_argument('--erm_amount', type=float, default=1.0)
parser.add_argument('--thre_init', type=float, default=-10000.0)
parser.add_argument('--early_loss_mean', type=str2bool, default=True)

parser.add_argument('--rex', type=str2bool, default=True)
parser.add_argument('--mse', type=str2bool, default=False)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--plot', type=str2bool, default=False)
parser.add_argument('--save_numpy_log', type=str2bool, default=False)
parser.add_argument('--clamp_max', type=float, default=1000.0)
parser.add_argument('--lbd', type=float, default=0.1)
parser.add_argument('--target_remain_rate', type=float, default=0.65)
args = parser.parse_args()

print('args:')
for k,v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))
  logging.info("\t{}: {}".format(k, v))
num_batches = (args.train_set_size // 2) // args.batch_size
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

  
  
color_dict = {'0': [255,0,0], '1': [255,255,0], '2': [0,255,0], '3': [0,100,0], '4': [0,0,255], '5': [255, 0,255],'6': [0,0,128], '7': [220,220,220], '8': [255,255,255], '9': [0,255,255]}
num_classes = 10  
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
  
train_data = torchvision.datasets.MNIST(root='/data/wyc', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
val_data = torchvision.datasets.MNIST(root='/data/wyc',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)
test_data = torchvision.datasets.MNIST(root='/data/wyc',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)

                                 
mnist_train = (train_data.data[:40000], train_data.targets[:40000])
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
# Define and instantiate the model

# Define loss function helpers
def updateBN(w, sparsity,mlp):
  bn_modules = list(filter(lambda m: (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)), mlp.named_children()))
  for m in bn_modules:
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
          m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))
# additional subgradient descent on the sparsity-induced penalty term
def updateGrad(w, sparsity, mlp):  
  bn_modules = list(filter(lambda m: (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)), mlp.named_children()))
  for m in bn_modules:
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        for i in range(w.shape[0]):
            m.weight.grad.data.add_(sparsity * w[i].unsqueeze(-1) * m.weight.data)
      
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
  scalar_criterion = nn.CrossEntropyLoss()
  loss =scalar_criterion(logits * scale, y.squeeze())
  grad = autograd.grad(loss, [scale], create_graph=True)[0]
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
model = maskvgg11(10)
#model=resnet.resnet20(10,args)
#model = torch.load("/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/resnet.pth.tar")
if use_cuda:
  mlp = model.cuda()
else:
  mlp = model

criterion = nn.CrossEntropyLoss(reduction='none')

optimizer = optim.SGD(mlp.parameters(), lr=args.lr, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, args.scheduler, gamma=0.1,last_epoch=-1)
pretty_print('step', 'train nll', 'train acc', 'rex penalty', 'irmv1 penalty', 'test acc')
train_los_pre=None
for epoch in range(args.epochs):
  loss_ce_list=[]
  i = 0
  highest_test_acc = 0.0

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
      #logits = mlp(env['images'][int(n*batch_size):int((n+1)*batch_size)])
      logits,env['_mask_list'],env['lasso_list'],env['_mask_before_list'],env['_avg_fea_list']= mlp(env['images'][int(n*batch_size):int((n+1)*batch_size)])
      env['nll'] = mean_nll(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
      env['acc'] = mean_accuracy(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
      env['penalty'] = penalty(logits, env['labels'][int(n*batch_size):int((n+1)*batch_size)])
      
    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    irmv1_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
    for l in range(len(env['_mask_list'])):
      _mask_list.append(torch.cat([envs[0]['_mask_list'][l], envs[1]['_mask_list'][l]],dim=0))
      lasso_list.append(torch.cat([envs[0]['lasso_list'][l], envs[1]['lasso_list'][l]],dim=0))
      _mask_before_list.append(torch.cat([envs[0]['_mask_before_list'][l], envs[1]['_mask_before_list'][l]],dim=0))
      _avg_fea_list.append(torch.cat([envs[0]['_avg_fea_list'][l], envs[1]['_avg_fea_list'][l]],dim=0))
    if use_cuda:
      weight_norm = torch.tensor(0.).cuda()
    else:
      weight_norm = torch.tensor(0.)
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)

    loss1 = envs[0]['nll']
    loss2 = envs[1]['nll']
    loss_ce_list.append(loss1.data.cpu())
    loss_ce_list.append(loss2.data.cpu())
    loss_each = torch.cat([loss1, loss2],dim=0)
    loss_ce_list.append(loss_each.data.cpu())    
    loss = 0.0
    loss += args.erm_amount * loss_each.mean()
   
    loss += args.l2_regularizer_weight * weight_norm
   
    #********************************  
    if train_los_pre != None:
      train_los_pre=train_los_pre.mean().cuda()
      w1=(loss_each < args.thre_cls*train_los_pre).float() 
      w2=((args.thre_cls*train_los_pre-loss_each)/(args.thre_cls*train_los_pre))      
      w=w1*w2                 
      for igraph in range(len(_avg_fea_list)):
        loss += (_avg_fea_list[igraph] * w).mean()

    #********************************
    optimizer.zero_grad()
    loss.backward()
    #updateBN(w,args.lbd, mlp)

    optimizer.step()
    scheduler.step()
  train_los_pre=torch.cat(loss_ce_list,dim=0)
  if epoch % args.eval_interval == 0:
    mlp.eval()
    with torch.no_grad():
      #logits = mlp(envs[2]['images'])
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
        #rex_penalty.detach().cpu().numpy(),
        #irmv1_penalty.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )
      logging.info("epoch: [{}]\t"
           "Train Loss {train_nll:.3f}\t"
           "Train Acc@1 {train_acc:.3f}\t"
           #"rex_penalty@1 {rex_penalty:.3f}\t"
           #"irmv1_penalty@1 {irmv1_penalty:.3f}\t"
           "test_acc Acc@1 {test_acc:.3f}\t"
           #.format(epoch, train_nll = train_nll, train_acc=train_acc, rex_penalty=rex_penalty, irmv1_penalty=irmv1_penalty, test_acc=test_acc))
           .format(epoch, train_nll = train_nll, train_acc=train_acc, test_acc=test_acc))

torch.save(mlp,'/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/vgg.pth.tar')

print('highest test acc this run:', highest_test_acc)
logging.info('highest test acc this run: {}'.format(highest_test_acc))

final_train_accs.append(train_acc.detach().cpu().numpy())
final_test_accs.append(test_acc.detach().cpu().numpy())
highest_test_accs.append(highest_test_acc)
print('Final train acc (mean/std across on epoch {} so far):')
print(np.mean(final_train_accs), np.std(final_train_accs))
logging.info('Final train acc (mean/std across restarts so far): {} / {}'.format(epoch,np.mean(final_train_accs), np.std(final_train_accs)))

print('Final test acc (mean/std across on epoch {} so far):')
print(np.mean(final_test_accs), np.std(final_test_accs))
logging.info('Final test acc (mean/std across restarts so far): {} / {}'.format(np.mean(final_test_accs), np.std(final_test_accs)))

print('Highest test acc (mean/std across on epoch {} so far):')
print(np.mean(highest_test_accs), np.std(highest_test_accs))
logging.info('Highest test acc (mean/std across restarts so far): {} / {}'.format(np.mean(highest_test_accs), np.std(highest_test_accs)))
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