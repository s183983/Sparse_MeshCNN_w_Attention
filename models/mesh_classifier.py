import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
import numpy as np
import os


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None
        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)
        self.criterion2 = networks.MRF_loss().to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).float()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation':
            self.labels = self.labels.long()       
       
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        '''one_ring = torch.from_numpy(self.mesh[0].gemm_edges)
        self.CE_loss = self.criterion(out, self.labels)
        self.prior_loss = self.opt.prior * self.criterion2(out, self.labels, one_ring)
        self.loss = self.CE_loss +self.prior_loss'''
        
        self.loss = self.criterion(out, self.labels)
        
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    
                    
                
        
##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)
            
    def save_network_acc(self, which_epoch,acc):
        """save model to disk"""
        save_filename = '%s_0%s_net.pth' % (which_epoch, round(acc*10000))
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self, loss_bool=False):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            
            label_class = self.labels
            if self.opt.dataset_mode == 'distance_field':
                correct = self.get_accuracy(out, label_class)
                dice = self.RMSE(out,label_class)
                self.export_segmentation(out.cpu())
                if loss_bool:
                    loss = self.get_loss(out)
                    return correct, len(label_class), dice, loss
                else:
                    return correct, len(label_class), dice

            else: 
                pred_class = out.data.max(1)[1]
                for i, m in enumerate(self.mesh):
                    filepath = os.path.join(m.export_folder, m.filename[:-4].split('\\')[-1])
                    #np.save(filepath, out.data[i,:,:].cpu().numpy())
                self.export_segmentation(pred_class.cpu())
                correct = self.get_accuracy(pred_class, label_class)
                dice = self.dice_score(pred_class, label_class)
                if loss_bool:
                    loss = self.get_loss(out)
                    return correct, len(label_class), dice, loss
                else:
                    return correct, len(label_class), dice

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        elif self.opt.dataset_mode == 'distance_field':
            #MAPE
            #correct = abs((pred-labels)/(labels+1e-10)).sum()/len(labels[0])
            #WMAPE
            correct = 0
            for i, mesh in enumerate(self.mesh):
                
                pred_i = pred[i,mesh.edges_count].cpu()
                label_i =labels[i,:mesh.edges_count].cpu()
                correct += (abs(pred_i-label_i)).sum()/abs(label_i).sum()
            correct /= len(self.mesh)
            #MSE
            #correct = ((pred-labels)**2).sum()/len(labels)
            
        return correct
    
    def dice_score(self, pred, labels):
        dice_sum = 0
        smooth=1e-6
        #correct_mat = labels.gather(2, pred.cpu().unsqueeze(dim=2))
        for i, mesh in enumerate(self.mesh):
            dice = 0
            pred_i = pred[i,:mesh.edges_count].cpu()
            label_i =labels[i,:mesh.edges_count].cpu()
            for idx in range(self.nclasses):
                pr = (pred_i==idx).float()
                lab =(label_i==idx).float()
                dice += ( 2*(pr*lab).sum() + smooth) / ( pr.sum()+lab.sum() +smooth)
            dice_sum += dice / self.nclasses
            
        return dice_sum/len(self.mesh)
    def RMSE(self, pred, labels):
        return torch.sqrt(((pred - labels)**2).mean())
    
    def get_loss(self, out):
        #one_ring = torch.from_numpy(self.mesh[0].gemm_edges)
        #CE_loss = self.criterion(out, self.labels)
        #prior_loss = self.opt.prior * self.criterion2(out, self.labels, one_ring)
        #loss = CE_loss + prior_loss
        loss = self.criterion(out, self.labels)
        return loss

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
                
        elif self.opt.dataset_mode == 'distance_field':
            if self.opt.export_folder:
                for mesh in self.mesh:
                    filename, file_extension = os.path.splitext(mesh.filename)
                    import platform
                    if platform.system()=="Windows":
                        file = '%s/%s_%d' % (mesh.export_folder, filename.split('\\')[-1], 0)
                    elif platform.system()=="Linux":
                        file = '%s/%s_%d' % (mesh.export_folder, filename.split('/')[-1], 0)
                    pred = pred_seg.cpu().numpy()
                    np.savez(file, edge_labels=pred)
        
