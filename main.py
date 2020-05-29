import utils
import data


def main(args):
    logger, result_dir, dir_name = utils.config_backup_get_log(args,__file__)
    device = utils.get_device()
    utils.set_seed(args.seed, device)

    # load data
    #  
    XY = data.Normal(args.dim, args.rho, device)
    




if __name__ == '__main__':
    args = utils.process_args()
    main(args)