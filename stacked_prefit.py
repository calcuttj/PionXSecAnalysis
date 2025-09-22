import ROOT as RT
from array import array
#import sys

from argparse import ArgumentParser as ap

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', type=str, required=True)
  parser.add_argument('-o', type=str, default='stacked_try.root')
  parser.add_argument('--no_upstream', action='store_true')
  parser.add_argument('--new_fv', action='store_true')
  parser.add_argument('--no_michel', action='store_true')
  parser.add_argument('--notitle', action='store_true')
  args = parser.parse_args()
  
  RT.gROOT.LoadMacro("~/protoDUNEStyle.C")
  RT.gROOT.SetStyle("protoDUNEStyle")
  RT.gROOT.ForceStyle()
  RT.gStyle.SetTitleX(0.5)
  RT.gStyle.SetTitleAlign(22)
  RT.gStyle.SetTitleY(0.87)
  RT.gStyle.SetTitleW(0.80) # or .85
  RT.gStyle.SetOptFit(111)

  # RT.gStyle.SetOptStat(00000)
  # RT.gStyle.SetErrorX(1.e-4)
  # RT.gStyle.SetTitleAlign(33)
  # RT.gStyle.SetTitleX(.915)
  # RT.gStyle.SetTitleY(.95)
  # RT.gROOT.ForceStyle()
  
  fIn = RT.TFile(args.i, "OPEN");
  
  
  the_dir = fIn.Get('MC_Samples')
  all_names = [i.GetName() for i in the_dir.GetListOfKeys()]
  hist_types = ['Abs', 'Cex', 'RejectedInt', 'APA2', 'MichelCut', 'FailedBeamCuts', 'NoBeamTrack']
  #hist_types = ['Abs', 'Cex', 'RejectedInt', 'APA2']
  samples = ['Abs', 'Cex', 'OtherInel', 'Muons', 'UpstreamIntSingle', 'PionPastFV', 'Other']
  if args.no_upstream: samples[4] = 'UpstreamInt'
  
  sample_names = ['Absorption', 'Charge Exchange', 'Other Interactions', 'Muons',
                  'Upstream Interactions', 'Pions Past FV', 'Decaying Pions']
  
  to_combine = ['NoBeamTrack', 'FailedBeamCuts', 'MichelCut']

  if args.no_michel:
    hist_types.pop(hist_types.index('MichelCut'))
    to_combine.pop(to_combine.index('MichelCut'))
    print(hist_types)
    print(to_combine)

  if args.new_fv:
    hist_types.append('BeforeFV')
    to_combine.insert(1, 'BeforeFV')
  
  colors = {
    'Abs':602,
    'Cex':433,
    'OtherInel':837,
    'Muons':419,
    'UpstreamIntSingle':403,
    'UpstreamInt':403,
    'PionPastFV':795,
    'Other':630
  }

  fills = {
    'Abs':1001,
    'Cex':3006,
    'OtherInel':3144,
    'Muons':3003,
    'UpstreamIntSingle':3004,
    'UpstreamInt':3004,
    'PionPastFV':3001,
    'Other':3008
  }
  
  sorted_hists = {
  }
  stacks = {}
  for t in hist_types:
    temp_dict = {}
    for s in samples:
      temp_dict[s] = []
    sorted_hists[t] = temp_dict
    ##if t not in to_combine:
    stacks[t] = RT.THStack() ##move in if uncomment above
  stacks['Combined'] = RT.THStack()
  
  for n in all_names:
    h = n.split('_')[-3]
    s = n.split('_')[1].replace('Underflow', '').replace('Overflow', '')
    #print(s, h)
    sorted_hists[h][s].append(fIn.Get('MC_Samples/%s'%n))
    #print(sorted_hists[h][s])
  

  for n, samps in sorted_hists.items():
    #if n in to_combine: continue
    #print(n, samps)
    for s, hists in samps.items():
      #print(s, hists)
      summed_hist = hists[0].Clone(s+'summed')
      summed_hist.Reset()
      for h in hists:
        #print(h)
        h.SetFillColor(colors[s])
        h.SetFillStyle(fills[s])

        h.SetLineColor(colors[s])
        h.Scale(1.e-3)
        if n == 'MichelCut': h.GetXaxis().SetBinLabel(1, '')
        # stacks[n].Add(h)
        summed_hist.Add(h)
      summed_hist.SetFillColor(colors[s])
      summed_hist.SetFillStyle(fills[s])
      summed_hist.SetLineColor(colors[s])
      summed_hist.SetLineWidth(1)
      stacks[n].Add(summed_hist)
  
  combined_hists = dict()
  combined_labels = {
    'MichelCut': 'Michel Vertex Cut',
    'NoBeamTrack': 'No Beam Track',
    'FailedBeamCuts': 'Failed Beam Cuts', 
    'BeforeFV': 'Before FV',
  }
  leg = RT.TLegend(.2, .55, .8, .83)
  leg.SetLineWidth(0)
  leg.SetFillStyle(0)
  for s,n in zip(samples, sample_names):
    combined_hists[s] = RT.TH1D('combined_%s'%s, '', len(to_combine), 0, len(to_combine))
    combined_hists[s].SetFillColor(colors[s])
    combined_hists[s].SetFillStyle(fills[s])
    combined_hists[s].SetLineColor(colors[s])
    for i in range(len(to_combine)):
      hists = sorted_hists[to_combine[i]][s]
      combined_hists[s].GetXaxis().SetBinLabel(i+1, combined_labels[to_combine[i]])
      for h in hists: combined_hists[s].AddBinContent(i+1, h.GetBinContent(1))
    #combined_hists[s].GetXaxis().SetBinLabel(1, 'Failed Beam Cuts')
    #combined_hists[s].GetXaxis().SetBinLabel(2, 'No Beam Track')
    #combined_hists[s].GetXaxis().SetBinLabel(3, 'Michel Vertex Cut')
    leg.AddEntry(combined_hists[s], n, 'lf')
    stacks['Combined'].Add(combined_hists[s]) 
  fOut = RT.TFile(args.o, 'recreate')
  
  xytitles = {
    'Abs':'Absorption Cand.;Reconstructed KE [MeV];Events per bin [x10^{3}]',
    'Cex':'Charge Exchange Cand.;Reconstructed KE [MeV];Events per bin [x10^{3}]',
    'RejectedInt':'Other Interaction Cand.;Reconstructed KE [MeV];Events per bin [x10^{3}]',
    'APA2':'Past Fiducial Volume;Reconstructed End Z [cm];Events per bin [x10^{3}]',
    'Combined':';;Events per bin [x10^{3}]',
    'MichelCut':'Michel Cut;;Events per bin [x10^{3}]',
    'FailedBeamCuts':';;Events per bin [x10^{3}]',
    'NoBeamTrack':';;Events per bin [x10^{3}]',
    'BeforeFV':';;Events per bin [x10^{3}]',
  }
  if args.notitle:
    xytitles = {k:';' + ';'.join(t.split(';')[1:]) for k, t in xytitles.items()}
    
  
  tt = RT.TLatex();
  tt.SetNDC();
  
  add_data = True #(len(sys.argv) > 2)
  if add_data:
    print('adding data')
    data_hists = {h:fIn.Get('Data/Data_selected_%s_hist'%h) for h in hist_types}
    print(data_hists)
    combined_errs = [data_hists[c].GetBinError(1) for c in to_combine]
    for h in data_hists.values():
      h.Sumw2()
      h.Scale(1e-3)
      h.SetMarkerStyle(20);
      h.SetMarkerColor(RT.kBlack);
    data_hists['Combined'] = RT.TH1D('combined_data', '', len(to_combine), 0, len(to_combine))
    for i in range(0, len(to_combine)):
      data_hists['Combined'].SetBinContent(i+1, data_hists[to_combine[i]].GetBinContent(1))
    data_hists['Combined'].Sumw2()
    data_hists['Combined'].SetMarkerColor(RT.kBlack);
    data_hists['Combined'].SetMarkerStyle(20);
    for i in range(len(combined_errs)): data_hists['Combined'].SetBinError(i+1, combined_errs[i]*1.e-3)
  
     
  
  for n, s in stacks.items():
    cname = f'c{n}'
    c = RT.TCanvas(cname, '')
    c.SetTicks()
    s.Draw('hist')
    #s.GetHistogram().SetTitle(xytitles[n])
    s.SetTitle(xytitles[n])
    s.GetHistogram().GetXaxis().CenterTitle()
    s.GetHistogram().GetYaxis().CenterTitle()
    # s.GetHistogram().GetYaxis().SetTitleOffset(.95)
    the_max = s.GetHistogram().GetMaximum()
    if add_data:
      if data_hists[n].GetMaximum() > s.GetHistogram().GetMaximum():
        print(data_hists[n].GetMaximum(), s.GetHistogram().GetMaximum())
        the_max = data_hists[n].GetMaximum()
    s.SetMaximum(2*the_max)
    s.Draw('hist')
    if add_data:
      data_hists[n].Draw('same e')
      if n == 'Abs':
        leg.AddEntry(data_hists[n], 'Data', 'pe')
    if n == 'Abs':
      leg.SetNColumns(2)
      leg.Draw('same')
    RT.gPad.RedrawAxis()
    tt.DrawLatex(0.10,0.94,"#bf{DUNE:ProtoDUNE-SP}");
    s.Write('stack_%s'%n)
    c.Write()
    c.SaveAs(cname + '.pdf')
    c.SaveAs(cname + '.png')
