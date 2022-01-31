import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test_script import run_test
import os
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_mat = []
        CE_mat = []
        prior_mat = []

        for i, data in enumerate(dataset):
            
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            loss_mat.append(model.loss.cpu().data.numpy())
            #CE_mat.append(model.CE_loss.cpu().data.numpy())
            #prior_mat.append(model.prior_loss.cpu().data.numpy())

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        writer.save_losses(loss_mat, CE_mat, prior_mat,epoch)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)
            

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)
            if acc >  0.9315:
                model.save_network_acc(epoch,acc)
            writer.plot_acc(acc, epoch)
    writer.close()
"""
debugfile('C:/Users/lowes/OneDrive/Skrivebord/DTU/6_Semester/Bachelor/MeshCNN_sparse/train.py',
          wdir='C:/Users/lowes/OneDrive/Skrivebord/DTU/6_Semester/Bachelor/MeshCNN_sparse', 
          args='--dataroot datasets/LAA_alligned/one_mesh --name geo_debug --arch meshdistance --dataset_mode distance_field --ncf 32 64 128 256 512 --ninput_edges 67100 --pool_res 24000 20000 16000 12000  --clamp 1.05 --features dist_xyz_curvature008_curvature004 
          num_threads 0' )
"""