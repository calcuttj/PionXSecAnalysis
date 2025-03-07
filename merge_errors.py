import ROOT as RT
from argparse import ArgumentParser as ap

if __name__ == '__main__':
    parser = ap()
    parser.add_argument('-i', type=str, nargs='+')
    parser.add_argument('-o', type=str)
    parser.add_argument('--nbins', type=int, default=9)
    args = parser.parse_args()

    print()

    h = RT.TH2D('xsec_cov', '', args.nbins, 0, args.nbins,
                args.nbins, 0, args.nbins)

    for part in args.i:
        f, name = part.split(',')
        print(f, name)
        fIn = RT.TFile.Open(f)
        hIn = fIn.Get(name)
        h.Add(hIn)
        # for i in range(0, args.nbins):
        #     for j in range(0, args.nbins):
        #         h.AddBinContent(i+1, j+1, h.GetBinContent(i+1, j+1))
    fOut = RT.TFile(args.o, 'recreate')
    h.Write()
    fOut.Close()
    