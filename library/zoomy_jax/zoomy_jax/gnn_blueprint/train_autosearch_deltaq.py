import argparse
from pathlib import Path
import itertools
import csv

try:
    from .train_deltaq import train
except ImportError:
    import sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from train_deltaq import train


def main():
    parser = argparse.ArgumentParser(description='Auto-search architecture/hyperparams for deltaQ model')
    parser.add_argument('--dataset', type=Path, default=Path('outputs/gnn_blueprint/dataset_deltaq.npz'))
    parser.add_argument('--out-root', type=Path, default=Path('outputs/gnn_blueprint/model_search'))
    parser.add_argument('--epochs-list', nargs='*', type=int, default=[80, 120, 180])
    parser.add_argument('--lr-list', nargs='*', type=float, default=[0.02, 0.01, 0.005])
    parser.add_argument('--variants', nargs='*', default=['multilevel_flow', 'linear_msg_edge'])
    parser.add_argument('--message-steps-list', nargs='*', type=int, default=[2, 3, 4, 6])
    parser.add_argument('--inner-iters-list', nargs='*', type=int, default=[1, 2, 3])
    parser.add_argument('--coarsen-levels-list', nargs='*', type=int, default=[2, 3, 4])
    parser.add_argument('--flow-modes', nargs='*', default=['tb', 'bt', 'bidir', 'alternating'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else (Path.cwd() / args.dataset)
    out_root = args.out_root if args.out_root.is_absolute() else (Path.cwd() / args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    combos = itertools.product(
        args.variants,
        args.epochs_list,
        args.lr_list,
        args.message_steps_list,
        args.inner_iters_list,
        args.coarsen_levels_list,
        args.flow_modes,
    )

    for variant, epochs, lr, msteps, iiters, clevels, flow in combos:
        tag = f"{variant}_ep{epochs}_lr{lr}_ms{msteps}_it{iiters}_cl{clevels}_{flow}"
        out_dir = out_root / tag
        print(f"[search] {tag}")
        test_loss = train(
            dataset, out_dir, epochs, lr, variant, args.seed,
            message_steps=msteps,
            inner_iters=iiters,
            coarsen_levels=clevels,
            flow_mode=flow,
        )
        rows.append({
            'tag': tag,
            'variant': variant,
            'epochs': epochs,
            'lr': lr,
            'message_steps': msteps,
            'inner_iters': iiters,
            'coarsen_levels': clevels,
            'flow_mode': flow,
            'test_loss': test_loss,
        })

    rows = sorted(rows, key=lambda r: r['test_loss'])
    with (out_root / 'search_results.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['tag', 'variant', 'epochs', 'lr', 'message_steps', 'inner_iters', 'coarsen_levels', 'flow_mode', 'test_loss'],
        )
        w.writeheader()
        w.writerows(rows)

    best = rows[0]
    with (out_root / 'best_model.txt').open('w', encoding='utf-8') as f:
        f.write(str(best))

    print(f"best={best}")


if __name__ == '__main__':
    main()
