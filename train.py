
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt.train_imgroot = r"D:\SoftWare\ChengxuXiangmu\LG_Data\SeismicData\Train"
    opt.train_maskroot = r"D:\SoftWare\ChengxuXiangmu\LG_Data\SeismicData\mask_npy"
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0


    if opt.continue_train:

        checkpoint_paths = [
            f"D:/SoftWare/ChengxuXiangmu/LGNet-main/checkpoints/celebahq_LGNet/{opt.load_model}_net_G1.pth",
            f"D:/SoftWare/ChengxuXiangmu/LGNet-main/checkpoints/celebahq_LGNet/{opt.load_model}_net_G2.pth",
            f"D:/SoftWare/ChengxuXiangmu/LGNet-main/checkpoints/celebahq_LGNet/{opt.load_model}_net_D.pth",
            f"D:/SoftWare/ChengxuXiangmu/LGNet-main/checkpoints/celebahq_LGNet/{opt.load_model}_net_G3.pth"
        ]


        print(f"Loading models from {checkpoint_paths}")

        for idx, network_name in enumerate(['G1', 'G2', 'D', 'G3']):

            checkpoint_path = f"D:/SoftWare/ChengxuXiangmu/LGNet-main1/checkpoints/celebahq_LGNet/{opt.load_model}_net_{network_name}.pth"
            print(f"Loading {network_name} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)


            model.load_networks(opt.load_model)

    model.train()

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

