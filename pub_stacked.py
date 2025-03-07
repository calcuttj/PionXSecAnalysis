import dunestyle.root as dunestyle
import ROOT as RT
from argparse import ArgumentParser as ap


def prefit(mcname, outname=None, dataname=None):
    #Open the input MC
    fMC = RT.TFile.Open(mcname)

    the_dir = fMC.Get('MC_Samples')
    all_names = [i.GetName() for i in the_dir.GetListOfKeys()]
    hist_types = ['Abs', 'Cex', 'RejectedInt', 'APA2', 'FailedBeamCuts', 'NoBeamTrack']
    samples = ['Abs', 'Cex', 'OtherInel', 'Muons', 'UpstreamIntSingle', 'PionPastFV', 'Other']
    
    sample_names = ['Absorption', 'Charge Exchange', 'Other Interactions', 'Muons',
                    'Upstream Interactions', 'Pions Past FV', 'Decaying Pions']
    
    sorted_hists = {}
    stacks = {}
    for t in hist_types:
        temp_dict = {}
        for s in samples:
            temp_dict[s] = []
        sorted_hists[t] = temp_dict

        stacks[t] = RT.THStack() ##move in if uncomment above
  
    for n in all_names:
        h = n.split('_')[-3]
        s = n.split('_')[1].replace('Underflow', '').replace('Overflow', '')
        #print(s, h)
        sorted_hists[h][s].append(fMC.Get(f'MC_Samples/{n}'))
        #print(sorted_hists[h][s])
  
    colors = [dunestyle.colors.NextColor() for i in range(len(samples))]
    print('Colors:', colors)
    for n, samps in sorted_hists.items():
        for i, (s, hists) in enumerate(samps.items()):
            for h in hists:
                dunestyle.colors.NextColor
                h.SetFillColor(colors[i])
                h.SetFillStyle(1001)
                h.SetLineColor(colors[i])
                h.Scale(1.e-3)
                stacks[n].Add(h)

    if outname is not None:
        fOut = RT.TFile(outname, 'recreate')
        for n,s in stacks.items():
            s.Write(n)
        fOut.Close()
  
if __name__ == '__main__':
    parser = ap()
    parser.add_argument('-o', type=str, default=None)
    parser.add_argument('--mc', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('routine', type=str, choices=['prefit'])
    args = parser.parse_args()

    prefit(args.mc, outname=args.o, dataname=args.data)