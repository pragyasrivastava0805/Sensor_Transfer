
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import re
import warnings
from    learner import Learner
from    copy import deepcopy
from torch.autograd import Variable
from torchmeta.modules.module import MetaModule
from collections import OrderedDict

def get_subdict(params, key=None):
    if params is None:
        return None
    _children_modules_parameters_cache = dict()
    all_names = tuple(params.keys())
    if (key, all_names) not in _children_modules_parameters_cache:
        if key is None:
            self._children_modules_parameters_cache[(key, all_names)] = all_names

    else:
        key_escape = re.escape(key)
        key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))

        _children_modules_parameters_cache[(key, all_names)] = [
        key_re.sub(r'\1', k) for k in all_names if key_re.match(k) is not None]

        names = _children_modules_parameters_cache[(key, all_names)]
     

        return OrderedDict([(name, params[f'{key}.{name}']) for name in names])

def gradient_update_parameters(model,loss,params=None,step_size=0.5,first_order=False):

    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        

        self.net = Learner(args.num_classes, args.imgc)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def adapt(self, inputs, targets, num_adaptation_steps=1,step_size=0.5):

        params = None
        for step in range(num_adaptation_steps):
            logits = self.net(inputs,params)
            inner_loss = F.cross_entropy(logits, targets)
            self.net.zero_grad()
            params = gradient_update_parameters(self.net, inner_loss, step_size=step_size, params = params)

        return params


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q =0
        corrects = 0

        for i in range(task_num):

            params = self.adapt(x_spt[i], y_spt[i],num_adaptation_steps=self.update_step,step_size=self.update_lr)
            with torch.no_grad():
                logits_q = self.net(x_qry[i],params)
                loss = F.cross_entropy(logits_q, y_qry[i])
                losses_q +=loss

            


                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects = corrects + correct

          




        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q / task_num

        # optimize theta parameters
        loss_q =Variable(loss_q, requires_grad = True)
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = (corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = 0
        losses_q = 0

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        params = self.adapt(x_spt, y_spt,num_adaptation_steps=self.update_step,step_size=self.update_lr)

        with torch.no_grad():

            logits_q = self.net(x_qry,params)
            loss = F.cross_entropy(logits_q, y_qry)
            losses_q +=loss

            


            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects = corrects + correct


   


           


        del net

        accs = corrects/ querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
