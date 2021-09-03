from dataset import *
from glow import *
import argparse
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--depth', type=int, default=32, help='Depth of the network (cf Glow figure 2).')
parser.add_argument('--n_levels', type=int, default=3, help='Number of levels of of the network (cf Glow figure 2).')
parser.add_argument('--width', type=int, default=512, help='Dimension of the hidden layers.')
# training params
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')


def main(args):
    train_holder = get_data('CIFAR10', True, None, None, False)
    test_holder = get_data('CIFAR10', False, None, None, False)

    cls = [0, 1, 8, 9]
    alpha = 0.5
    need_svm_label = False
    norm_flag = False

    train_bin = train_holder.pos_neg_split(cls)
    test_bin = train_holder.pos_neg_split(cls)

    train_data, pi = train_bin.get_dataset(alpha, c=0.5, svm_labels=False)
    test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

    flow_p = Glow(width=args.width, depth=args.depth, n_levels=args.n_levels).to(device)
    flow_u = Glow(width=args.width, depth=args.depth, n_levels=args.n_levels).to(device)

    f = open('results.txt', 'w')

    flow_u.fit(train_data, batch_size=args.batch_size, num_epochs=args.num_epochs, lr=args.lr, outfile=f)
    flow_p.fit(train_data, batch_size=args.batch_size, num_epochs=args.num_epochs, lr=args.lr, outfile=f)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=512)

    y_true = np.array([])
    p_prob = np.array([])
    u_prob = np.array([])

    for (x, y, _) in test_loader:
        p_pred = flow_p.log_prob(x.to(device).float())
        u_pred = flow_u.log_prob(x.to(device).float())

        p_prob = np.hstack((p_prob, p_pred.squeeze().detach().cpu().numpy()))
        u_prob = np.hstack((u_prob, u_pred.squeeze().detach().cpu().numpy()))

        y_true = np.hstack((y_true, y.squeeze().detach().cpu().numpy()))
    print(f"full auc={metrics.roc_auc_score(y_true, p_prob - u_prob)}", file=f)
    print(f"pos auc={metrics.roc_auc_score(y_true, p_prob)}", file=f)
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
