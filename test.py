from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    dice_sum = 0
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, dice = model.test()
        dice_sum += dice
        writer.update_counter(ncorrect, nexamples)
        writer.save_test_acc(data, ncorrect, nexamples, dice)
    dice_sum /= len(dataset)
    writer.print_acc(epoch, writer.acc, dice_sum)
    return writer.acc


if __name__ == '__main__':
    run_test()
