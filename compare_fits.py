from argparse import ArgumentParser as ap
import numpy as np
import ROOT as RT
RT.gStyle.SetOptStat(0)
import yaml


def read_yaml(c):
  with open(c, 'r') as fin: config = yaml.safe_load(fin)
  return config

def get_sels(args, config, f1, fs, fOut):
  ranges = {
    'Abs': [0.] + [float(i) for i in range(400, 950, 50)] + [1200.],
    'Cex': [0.] + [float(i) for i in range(400, 950, 50)] + [1200.],
    'RejectedInt': [0.] + [float(i) for i in range(400, 950, 50)] + [1200.],
  }
  titles = {
    'Abs':'Selected Absorption;Reconstructed KE [MeV];Number of Entries [x10^{3}]',
    'Cex':'Selected Ch. Exch.;Reconstructed KE [MeV];Number of Entries [x10^{3}]',
    'RejectedInt':'Selected Other;Reconstructed KE [MeV];Number of Entries [x10^{3}]',
    'APA2': 'Past FV;;Number of Entries [x10^{3}]',
    'MichelCut': 'Michel Cut;;Number of Entries [x10^{3}]',
    'FailedBeamCuts': ';;Number of Entries [x10^{3}]',
    'NoBeamTrack': ';;Number of Entries [x10^{3}]',
  }
  dists = ['Abs', 'Cex', 'RejectedInt', 'APA2', 'MichelCut', 'FailedBeamCuts',
          'NoBeamTrack']
  if args.no_michel:
    dists.pop(dists.index('MichelCut'))
  results_hs = {}

  for d in dists:
    h = f1.Get(f'PostFit{d}Total')
    if d in ranges.keys():
      hnew = RT.TH1D('hnew_{d}', '', len(ranges[d])-1, np.array(ranges[d]))
      for i in range(1, h.GetNbinsX()+1):
        hnew.SetBinContent(i, h.GetBinContent(i))
      h = hnew
    else:
      h.GetXaxis().SetBinLabel(1, '')

    h.SetLineColor(RT.kBlack)
    h.SetDirectory(0)
    h.SetFillStyle(0)
    h.Scale(1.e-3)
    results_hs[d] = h

  hs = {}
  for d in dists:
    hs[d] = []
    for f in fs:
      if d in ranges.keys():
        h = f.Get(f'PostFit{d}Total')
        hnew = RT.TH1D('hnew_{d}', '', len(ranges[d])-1, np.array(ranges[d]))
        for i in range(1, h.GetNbinsX()+1):
          hnew.SetBinContent(i, h.GetBinContent(i))
        hs[d].append(hnew)
      else:
        hs[d].append(f.Get(f'PostFit{d}Total'))
      hs[d][-1].SetDirectory(0)
      hs[d][-1].SetFillStyle(0)
      hs[d][-1].SetLineColor(RT.kRed)
      hs[d][-1].Scale(1.e-3)


  fOut.cd()
  for d in dists:
    c = RT.TCanvas(f'c_{d}_sel')
    c.SetTicks()
    results_hs[d].SetMaximum(get_max(results_hs[d], hs[d]))
    results_hs[d].SetMinimum(0)
    results_hs[d].SetTitle(titles[d])
    set_title_atts(results_hs[d])
    results_hs[d].Draw('hist')
    for hi in hs[d]: hi.Draw('hist same')
    results_hs[d].Draw('hist same')

    if d == 'Abs':
      leg = RT.TLegend()
      leg.AddEntry(results_hs[d], 'Base Fit', 'l')
      leg.AddEntry(hs[d][0], 'Syst. Fits', 'l')
      leg.SetLineWidth(0)
      leg.SetFillStyle(0)
      leg.Draw()
      leg.Write('leg')

    c.Write()
    c.SaveAs(f'c_{d}_sel.pdf')

def get_from_file(f, name):
  h = f.Get(name)
  h.SetDirectory(0)
  h.SetFillStyle(0)
  return h

def set_title_atts(h):
  h.GetXaxis().SetTitleOffset(.8);
  h.GetYaxis().SetTitleOffset(.8);
  h.SetTitleSize(.05, "XY");


def get_xsecs(args, config, f1, fs, fOut):
  dists = ['Abs', 'Cex', 'OtherInel']

  true_ke = 'True KE [MeV]'
  entries = 'Number of Entries [x10^{3}]'
  titles = {
    'Abs': f'Absorption Interactions;{true_ke};{entries}',
    'Cex': f'Ch. Exch. Interactions;{true_ke};{entries}',
    'OtherInel': f'Other Interactions;{true_ke};{entries}',
  }

  xsecstr = '#sigma [mb]'
  xsec_titles = {
    'Abs': f'Absorption;{true_ke};{xsecstr}',
    'Cex': f'Ch. Exch.;{true_ke};{xsecstr}',
    'OtherInel': f'Other;{true_ke};{xsecstr}',
  }

  results_hs = {}
  results_hs_ints = {}
  for d in dists:
    results_hs[d] = get_from_file(f1, f'PostFitXSec/PostFit{d}XSec')
    results_hs_ints[d] = get_from_file(f1, f'PostFitXSec/PostFit{d}Hist')
    results_hs[d].SetLineColor(RT.kBlack)
    results_hs_ints[d].SetLineColor(RT.kBlack)
    results_hs_ints[d].SetTitle(titles[d])
    results_hs[d].SetTitle(xsec_titles[d])
    set_title_atts(results_hs[d])
    set_title_atts(results_hs_ints[d])


    results_hs_ints[d].Scale(1.e-3)

  hs = {}
  hs_ints = {}
  for d in dists:
    hs[d] = []
    hs_ints[d] = []
    for f in fs:
      hs[d].append(get_from_file(f, f'PostFitXSec/PostFit{d}XSec'))
      hs_ints[d].append(get_from_file(f, f'PostFitXSec/PostFit{d}Hist'))
      hs[d][-1].SetLineColor(RT.kRed)
      hs_ints[d][-1].SetLineColor(RT.kRed)
      hs_ints[d][-1].Scale(1.e-3)

  name = 'PostFitXSec/PostFitTotalIncidentAbsUnderflow'
  results_hs_inc = get_from_file(f1, name)
  results_hs_inc.SetLineColor(RT.kBlack)
  results_hs_inc.SetTitle( f'Incident Entries;{true_ke};Number of Entries [x10^6]')
  set_title_atts(results_hs_inc)
  results_hs_inc.Scale(1.e-6)
  hs_incs = [
    get_from_file(f, name)
    for f in fs
  ]
  for h in hs_incs:
    h.SetLineColor(RT.kRed)
    h.Scale(1.e-6)


  fOut.cd()
  for d in dists:
    c = RT.TCanvas(f'c_{d}_xsec')
    c.SetTicks()
    results_hs[d].SetMaximum(get_max(results_hs[d], hs[d]))
    results_hs[d].SetMinimum(0)
    results_hs[d].Draw('hist')
    for hi in hs[d]: hi.Draw('hist same')
    results_hs[d].Draw('hist same')
    c.Write()
    c.SaveAs(f'c_{d}_xsec.pdf')

    c = RT.TCanvas(f'c_{d}_int')
    c.SetTicks()
    results_hs_ints[d].SetMaximum(get_max(results_hs_ints[d], hs_ints[d]))
    results_hs_ints[d].SetMinimum(0)
    results_hs_ints[d].Draw('hist')
    for hi in hs_ints[d]: hi.Draw('hist same')
    results_hs_ints[d].Draw('hist same')
    c.Write()
    c.SaveAs(f'c_{d}_int.pdf')

  c = RT.TCanvas('c_inc')
  c.SetTicks()
  results_hs_inc.SetMaximum(get_max(results_hs_inc, hs_incs))
  results_hs_inc.SetMinimum(0)
  results_hs_inc.Draw('hist')
  for h in hs_incs: h.Draw('hist same')
  results_hs_inc.Draw('hist same')
  c.Write()
  c.SaveAs(f'c_inc.pdf')

def get_max(h, hs):
  return 1.1*max([h.GetMaximum()] + [hi.GetMaximum() for hi in hs])

def normal(args):
  config = read_yaml(args.c)

  fOut = RT.TFile(args.o, 'recreate')
  f1 = RT.TFile.Open(config['results'])
  fs = []
  for fname in config['systs']:
    fs.append(RT.TFile.Open(fname))

  get_sels(args, config, f1, fs, fOut)
  get_xsecs(args, config, f1, fs, fOut)
  fOut.Close()
  f1.Close()
  for f in fs:
    f.Close()


if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-c', type=str, help='yaml config file')
  parser.add_argument('-o', type=str)
  parser.add_argument('--no_michel', action='store_true')
  parser.add_argument('--routine', type=str, default=None,
                      choices=[
                        None,
                      ]
                     )
  args = parser.parse_args()

  routines = {
    None:normal,
  }

  routines[args.routine](args)
