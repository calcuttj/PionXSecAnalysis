import ROOT as RT
from argparse import ArgumentParser as ap
import numpy as np

if __name__ == '__main__':
    parser = ap()
    parser.add_argument('--thresh', type=str, required=True)
    parser.add_argument('--no-thresh', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--bins', type=float, nargs='+')
    args = parser.parse_args()

    fThresh = RT.TFile.Open(args.thresh)
    fNoThresh = RT.TFile.Open(args.no_thresh)

    abs_thresh = fThresh.Get('abs_KE')
    abs_no_thresh = fNoThresh.Get('abs_KE')

    xs = [
        np.arange(args.bins[i], args.bins[i+1], 1.)
        for i in range(len(args.bins)-1)
    ]
    print(xs)
    thresh_ys = [
        np.array([abs_thresh.Eval(x) for x in these_xs])
        for these_xs in xs
    ]
    no_thresh_ys = [
        np.array([abs_no_thresh.Eval(x) for x in these_xs])
        for these_xs in xs
    ]

    ratios = [np.mean(nt/t) for nt, t in zip(no_thresh_ys, thresh_ys)]
    print(ratios)

    fThresh.Close()
    fNoThresh.Close()
