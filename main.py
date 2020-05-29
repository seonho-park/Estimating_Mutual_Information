import math
import torch

import utils
import data
import methods


def main(args):
    logger, result_dir, dir_name = utils.config_backup_get_log(args,__file__)
    device = utils.get_device()
    utils.set_seed(args.seed, device)

    # load data
    pXY = data.Normal(args.dim, args.rho, device)
    model = methods.setup_method(args.method, args.dim, args.hidden, args.layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        X, Y = pXY.draw_samples(args.N)
        XY_package = torch.cat([X.repeat_interleave(X.size(0), 0), Y.repeat(Y.size(0), 1)], dim=1)
        optim.zero_grad()
        L = model(X, Y, XY_package)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim.step()
        print('step {:4d} | '.format(step), end='')
        print('ln N: {:.2f} | I(X,Y): {:.2f} | est. I(X,Y): {:.2f}'.format(math.log(args.N), pXY.I(), -L.item()))

    # Final evaluation
    M = args.N
    X, Y = pXY.draw_samples(M)
    XY_package = torch.cat([X.repeat_interleave(M, 0), Y.repeat(M, 1)], dim=1)
    test_MI = {}
    model.eval()
    test_MI = -model(X, Y, XY_package).item()
    print('{:6.2f}'.format(test_MI))
    print('ln({:d}): {:.2f} | I(X,Y): {:.2f}'.format(M, math.log(M), pXY.I()))
    

if __name__ == '__main__':
    args = utils.process_args()
    main(args)