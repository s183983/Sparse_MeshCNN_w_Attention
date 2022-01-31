import os
import time
import numpy as np

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None

class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.final_test_file = os.path.join(self.save_dir, 'final_test_acc.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0
        #
        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None
        if opt.phase =="final_test":
            with open(self.final_test_file, "a") as log_file:
                log_file.write('%s\n' % opt.phase)

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """ prints train loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                  % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)
            
    def save_losses(self, loss_mat, CE_loss, prior_loss,epoch):
        file  =os.path.join(self.save_dir,'train_losses.npz')
        if epoch==1:
            loss_load = loss_mat
            CE_load = CE_loss
            prior_load = prior_loss
        else:
            loaded = np.load(file)
            loss_load = loaded['loss']
            CE_load = loaded['CE_loss']
            prior_load = loaded['prior_loss']
            loss_load = np.vstack(( loss_load, np.asarray(loss_mat) ))
            CE_load = np.vstack(( CE_load, np.asarray(CE_loss) ))
            prior_load = np.vstack(( prior_load, np.asarray(prior_loss) ))
        
        np.savez(file, loss = loss_load, CE_loss = CE_load, prior_loss = prior_load)
    
    def save_val_loss(self, loss, epoch):
        file = os.path.join(self.save_dir,'val_loss.npz')
        if epoch==1:
            loss_load = loss
        else:
            loaded = np.load(file)
            loss_load = loaded['val_loss']
            loss_load = np.vstack(( loss_load, np.asarray(loss) ))
        
        np.savez(file, val_loss = loss_load)
        
    def save_test_acc(self, data, ncorrect, nexamples, dice):
        _, name = os.path.split(data['mesh'][0].filename)
        
        if self.opt.dataset_mode == 'distance_field':
            message = 'File: {}, Accuracy: [{:.4} %], RMSE: [{:.4} %]' \
                .format( name, ncorrect/nexamples*100, dice)
        else:
            message = 'File: {}, Accuracy: [{:.4} %], Dice score: [{:.4} %]' \
                .format( name, ncorrect/nexamples*100, dice*100)
        
        with open(self.final_test_file, "a") as log_file:
            log_file.write('%s\n' % message)
        
    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def print_acc(self, epoch, acc, dice):
        """ prints test accuracy to terminal / file """
        if self.opt.dataset_mode == 'distance_field':
            message = 'epoch: {}, TEST ACC: [{:.5} %], RMSE: [{:.5} ]\n' \
                .format(epoch, acc * 100, dice)
        else:
            message = 'epoch: {}, TEST ACC: [{:.5} %], Dice score: [{:.5} %]\n' \
                .format(epoch, acc * 100, dice*100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
