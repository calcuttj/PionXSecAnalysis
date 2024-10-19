import ROOT as RT
RT.gROOT.SetBatch()
import sys
from argparse import ArgumentParser as ap
import matplotlib.pyplot as plt
import numpy as np

tt = RT.TLatex();
tt.SetNDC();
titles = {
  'abs': 'Absorption',
  'cex': 'Ch. Exchange',
  'other': 'Other',
}


class DataHolder:
  def __init__(self):
    self.xsecs = {
      'abs':[],
      'cex':[],
      'other':[]
    }
    self.fakes = {
      'abs':[],
      'cex':[],
      'other':[]
    }

    self.ranges = None
    self.parameters = []

def open_list(i):
  with open(i, 'r') as f:
    lines = f.readlines()

  lines = [l.replace('/pnfs', 'root://fndca1.fnal.gov:1094//pnfs/fnal.gov/usr').strip() for l in lines]
  return lines

def get_ranges(args, f):
  fIn = RT.TFile.Open(f)
  ranges = dict()
  for n,t in [('abs', 'Abs'),
              ('cex', 'Cex'),
              ('other', 'OtherInel')]:
    name = f'{t}Hist' if args.ints else f'{t}XSec'
    hname = f'PreFitXSec/PreFit{name}'
    h = fIn.Get(hname)
    ranges[n] = ((
      h.GetBinLowEdge(1),
      h.GetBinLowEdge(h.GetNbinsX()+1)))
  fIn.Close()
  return ranges

def get_xsecs(args, f):
  fIn = RT.TFile.Open(f)

  hesse = fIn.Get('post_hesse_cov_status')
  if not args.nocheck and 3 not in [i for i in hesse]:
    print('warning', f)
    return (None, None, None)

  xsecs = dict()
  fakes = dict()
  ranges = dict()
  for n,t in [('abs', 'Abs'),
              ('cex', 'Cex'),
              ('other', 'OtherInel')]:
    name = f'{t}Hist' if args.ints else f'{t}XSec'
    hname = f'PostFitXSec/PostFit{name}'
    h = fIn.Get(hname)

    if not args.throws:
      xsecs[n] = np.array([h.GetBinContent(i+1) for i in range(h.GetNbinsX())])
    else:
      gr = fIn.Get(f'Throws/grXSecThrow{t}Underflow')
      xsecs[n] = np.array(gr.GetY())
    if n == 'abs': print(f, xsecs[n][0])
    ranges[n] = ((
      h.GetBinLowEdge(1),
      h.GetBinLowEdge(h.GetNbinsX()+1)))

    name = 'Tot' if args.ints else 'XSec'
    h = fIn.Get(f'FakeDataXSecs/FakeData{t}Fake{name}')
    fakes[n] = np.array([h.GetBinContent(i+1) for i in range(h.GetNbinsX())]) if not args.nofake else None 

  h = fIn.Get('postFitParsNormal')
  pars = [h.GetBinContent(i) for i in range(1, h.GetNbinsX()+1)]

  fIn.Close()
  return xsecs, ranges, fakes, pars

def get_g4_xsec(fG4, name):
  #scales = np.array(scales)
  #if name in ['abs', 'cex']:
    g = fG4.Get(f'{name}_KE')
    return g
  #else:
  #  gs = [
  #    fG4.Get(f'{name}_KE') for name in ['dcex', 'inel', 'prod']
  #  ]
  #  ys = np.array(gs[0].GetY())
  #  ys += np.array(gs[1].GetY())
  #  ys += np.array(gs[2].GetY())

  #  xs = np.array(gs[0].GetX())
  #  g = RT.TGraph(len(xs), xs, ys)
  #  return g



def get_g4_xsecs_config(fG4):
  #fG4 = RT.TFile.Open(config['g4_file'])
  gs = {n:get_g4_xsec(fG4, n) for n in ['abs', 'cex', 'other']}
  a = 0
  #for n, g in gs.items():
    #ys = np.array(g.GetY())
    #xs = np.array(g.GetX())
    #gs[n] = RT.TGraph(len(xs), xs, ys)
    #a += 1
    #g.SetDirectory(0)
  #fG4.Close()
  return gs

def get_g4_xsecs(args):
  fG4 = RT.TFile.Open(args.g4)
  gs = {n:get_g4_xsec(fG4, n) for n in ['abs', 'cex', 'other']}
  a = 0
  for n, g in gs.items():
    ys = args.scales[a]*np.array(g.GetY())
    xs = np.array(g.GetX())
    gs[n] = RT.TGraph(len(xs), xs, ys)
    a += 1
  fG4.Close()
  return gs
  

def plot_xsecs_root(means, cov, xsecs, fakes, fOut, ranges, args):
  print(ranges)

  titles = {
    'abs': 'Absorption',
    'cex': 'Ch. Exchange',
    'other': 'Other',
  }

  gs = get_g4_xsecs(args)
  #hPulls = dict()
  #for n in titles.keys():
  #  for i in range(26):
  #    hPulls[(n, i)] = n:RT.TH1D(f'hPulls{n}', '', 100, -10, 10) for n in titles.keys()
  nbins = sum([len(m) for m in means.values()])
  hPulls2D = RT.TH2D('hPulls2D', ';Cross Section Bin;r', nbins, 0, nbins, 100, -1, 1)
  hPulls2D.SetTitleOffset(.6, "XY")
  hPulls2D.SetTitleSize(.06, "XY")
  hVals2D = RT.TH2D('hVals2D', '', nbins, 0, nbins, 1000, 0, 1000)
  fOut.cd()
  g4_vals = dict()
  a = 0
  ey = []
  for n, m in means.items():
    print(len(m))
    h = RT.TH1D(f'h{n}', '', len(m), *(ranges[n]))
    g4_vals[n] = []
    xs = [h.GetBinCenter(i+1) for i in range(len(m))]
    ys = [h.GetBinContent(i+1) for i in range(len(m))]
    for i in range(len(m)):
      h.SetBinContent(i+1, m[i])
      h.SetBinError(i+1, np.sqrt(cov[i, i]))
      ey.append(np.sqrt(cov[i, i]))
      x = h.GetBinCenter(i+1)
      g4 = gs[n].Eval(x)
      g4_vals[n].append(g4)
    h.Write()

    g4s = np.array(g4_vals[n])
    xsec_vals = xsecs[n]
    print(g4s, xsec_vals)
    if args.fake_diff:
      diffs = fakes[n] - xsec_vals
    else:
      diffs = g4s - xsec_vals
    #print('diffs', diffs)
    for i in range(len(m)):
      for v in xsec_vals[:, i]:
        hVals2D.Fill(a+.5, v)

      for j, d in enumerate(diffs[:, i]):
        p = d/fakes[n][j][i] if args.fake_diff else d/g4s[i]
        if a == 0 and p > .2: print(a, j)
        hPulls2D.Fill(a+.5, p)
      a += 1
  hPulls2D.Write()
  hVals2D.Write()
  for i in range(hPulls2D.GetNbinsX()):
    h = hPulls2D.ProjectionY(f'h{i}', i+1, i+1)
    h.GetYaxis().SetTitle('Number of Fits')
    h.SetTitleOffset(.6, "XY")
    h.SetTitleSize(.06, "XY")
    h.Fit('gaus')
    h.Write()

  for i in range(hVals2D.GetNbinsX()):
    h = hVals2D.ProjectionY(f'hVal{i}', i+1, i+1)
    h.Fit('gaus')
    h.Write()

  hCov = RT.TH2D('xsec_cov', '', len(cov), 0, len(cov),
                             len(cov), 0, len(cov))
  hCorr = RT.TH2D('xsec_corr', '', len(cov), 0, len(cov),
                                    len(cov), 0, len(cov))
  for i in range(len(cov)):
    for j in range(len(cov)):
      hCov.SetBinContent(i+1, j+1, cov[i, j])
      hCorr.SetBinContent(i+1, j+1, cov[i, j]/np.sqrt(cov[i, i]*cov[j, j]))
  hCov.Write()
  hCorr.Write()
def plot_xsecs(xsecs):
  print('plotting xsecs')

  lens = []
  for n, x in xsecs.items():
    lens.append(x.shape[1])

  n_toys = xsecs['abs'].shape[0]
  print(n_toys, sum(lens))
  all_xsecs = np.zeros((n_toys, sum(lens)))

  total = 0
  for n, x in xsecs.items():
    all_xsecs[:, total:total+x.shape[1]] = x
    print(total, total+x.shape[1])
    total += x.shape[1]
    #for i in range(x.shape[1]):
      #plt.hist(x[:, i], label='#sigma')
      #plt.savefig(f'g4rw_fluc_{n}_{i}.pdf')
      #plt.close()

  cov = np.cov(all_xsecs.T)
  #plt.imshow(cov)
  #plt.savefig('g4rw_fluc_cov.pdf')
  #plt.close()

  corr = np.corrcoef(all_xsecs.T)
  #plt.imshow(corr)
  #plt.savefig('g4rw_fluc_corr.pdf')
  #plt.close()

  means = {
    n: np.mean(xsecs[n], axis=0) for n in titles.keys()
  }
  print(means)
  return (means, cov)


def get_all_xsecs(args, lines):
  the_holder = DataHolder()

  a = 0
  for l in lines:
    if not a % 100: print(f'{a}/{len(lines)}', end='\r')
    xs, ranges, fs, pars = get_xsecs(args, l)
    if xs is None: continue
    for n, x in xs.items():
      the_holder.xsecs[n].append(x)
    for n, x in fs.items():
      the_holder.fakes[n].append(x)
    the_holder.parameters.append(pars)
    a += 1
  for n, x in the_holder.xsecs.items():
    the_holder.xsecs[n] = np.array(x)
  for n, x in the_holder.fakes.items():
    the_holder.fakes[n] = np.array(x)

  the_holder.ranges = get_ranges(args, lines[0])

  return the_holder

def get_chi2(args, f):
  fIn = RT.TFile.Open(f)
  hesse = fIn.Get('post_hesse_cov_status')
  if 3 not in [i for i in hesse]: return None
  chi2 = [fIn.Get('chi2_stat')[0], fIn.Get('chi2_syst')[0]]
  fIn.Close()
  return chi2
  
def get_all_chi2s(args, lines):
  a = 0
  all_chi2s = []
  for l in lines:
    if not a % 100: print(f'{a}/{len(lines)}', end='\r')
    chi2 = get_chi2(args, l)
    if chi2 is not None:
      all_chi2s.append(chi2)
    a += 1
  all_chi2s = np.array(all_chi2s)
  return all_chi2s

def exclude_vals(h_vals, g_vals, cov):
  g_vals['abs'] = g_vals['abs'][1:]
  h_vals = h_vals[1:]
  cov = cov[1:, 1:]
  return h_vals, g_vals, cov

def style_g4(g4, g4max, title):
  g4.SetMaximum(g4max)
  g4.GetXaxis().SetRangeUser(0., 999.)
  g4.SetLineColor(RT.kRed)
  g4.SetLineWidth(2)
  g4.SetTitle(f';Reconstructed KE (MeV);Cross Section (mb)')
  g4.GetXaxis().CenterTitle()
  g4.GetYaxis().CenterTitle()
  #g4.GetXaxis().SetTitleSize(.06)
  #g4.GetXaxis().SetTitleOffset(.6)
  #g4.GetYaxis().SetTitleSize(.06)
  #g4.GetYaxis().SetTitleOffset(.6)

def get_g4_max(g4):
  return max([y for y in g4.GetY()])

def plot_results(hs, g4s, cov):
  g4max = get_g4_max(g4s['abs']) #g4s['abs'].GetMaximum()
  print(g4max)
  add_leg = True
  for n, h in hs.items():
    xs = []
    ys = []
    eys = []
    for i in range(h.GetNbinsX()):
      h.SetBinError(i+1, np.sqrt(cov[i, i]))
      xs.append(h.GetBinCenter(i+1))
      ys.append(h.GetBinContent(i+1))
      eys.append(np.sqrt(cov[i, i]))

    c = RT.TCanvas(f'c_results_{n}')
    style_g4(g4s[n], g4max, titles[n])
    g4s[n].Draw('AC')
    g4s[n].SetMaximum(1.05*g4max)
    h.SetMarkerStyle(20)
    h.SetMarkerColor(RT.kBlack)
    h.SetLineColor(RT.kBlack)
    h.Draw('same pe1')
    if add_leg:
      leg = RT.TLegend()
      leg.AddEntry(g4s[n], 'Geant4 4.10.6', 'l')
      leg.AddEntry(h, 'ProtoDUNE-SP', 'lp')
      leg.SetBorderSize(0)
      leg.SetFillStyle(0)
      leg.Draw()
      add_leg = False
    tt.DrawLatex(0.10,0.94,"#bf{DUNE:ProtoDUNE-SP}");
    c.SetTicks()
    RT.gPad.RedrawAxis()
    c.SaveAs(f'c_results_{n}.pdf')
    c.Write()

    gr = RT.TGraphAsymmErrors(
      len(xs), np.array(xs), np.array(ys), np.array([0.]*len(xs)), np.array([0.]*len(xs)),
      np.array(eys), np.array(eys)
    )
    gr.Write(f'{n}_xsec')

  
def test_files(args):
  lines = open_list(args.i)
  bad_files = []
  good_files = []
  for f in lines:
    try:
      fIn = RT.TFile.Open(f)
      fIn.Close()
      good_files.append(f+'\n')
    except OSError:
      print('Found bad file', f)
      bad_files.append(f)
  print('\nBad files')
  for bf in bad_files: print(bf)
  print()
  with open(args.i, 'w') as f: f.writelines(good_files)

def plot_pars(pars):
  nbins = pars.shape[1]
  hPars2D = RT.TH2D('hPars2D', ';Parameter Number;Values', nbins, 0, nbins, 500, 0, 5)
  for l in pars:
    for i in range(len(l)):
      hPars2D.Fill(i+.5, l[i])
  hPars2D.Write()


def process(args):
  lines = open_list(args.i)
  #xsecs, ranges, fakes = get_all_xsecs(args, lines)
  results = get_all_xsecs(args, lines)
  xsecs = results.xsecs
  ranges = results.ranges
  fakes = results.fakes
  means, cov = plot_xsecs(xsecs)
  results.parameters = np.array(results.parameters)

  g4s = get_g4_xsecs(args)
  if args.results != None:
    hs = get_results(args)

  if args.fit:
    h_vals, g_vals = get_vals(args, hs, g4s)
    print(h_vals)
    print(g_vals)

    #fit_func(np.array([3,3,3]), h_vals, g_vals, cov)
    if args.exclude:
      h_vals, g_vals, cov = exclude_vals(h_vals, g_vals, cov)
    res = minimize(fit_func, x0=np.ones(3), args=(h_vals, g_vals, cov), method='Powell')
    print(res)

  fOut = RT.TFile(args.o, 'recreate')
  print('plotting')
  plot_xsecs_root(means, cov, xsecs, fakes, fOut, ranges, args)
  if args.results != None:
    plot_results(hs, g4s, cov)
  plot_pars(results.parameters)
  fOut.Close()

def fit_func(scales, h_vals, g_vals, cov):

  all_gs = np.zeros(len(h_vals))
  a = 0
  for s,g in zip(scales, g_vals.values()):
    all_gs[a:a+len(g)] = s*g
    a += len(g) 

  diffs = h_vals - all_gs
  chi2 = np.dot(diffs,
                np.dot(np.linalg.inv(cov), diffs))
  return chi2
  
    

def get_vals(args, hs, g4s):

  total = 0
  for h in hs.values():
    total += h.GetNbinsX()

  '''all_xs = np.zeros(total)
  a = 0
  for xi in xs.values():
    for x in xi:
      all_xs[a] = x
      a += 1'''

  if args.ave_g4:
    xs = dict()
    for n, h in hs.items():
      bin_xs = []
      xs[n] = [
        np.arange(h.GetBinLowEdge(i), h.GetBinLowEdge(i), .01)
        for i in range(1, h.GetNbinsX+1)
      ]

    g4_vals = {
      n:np.mean([[g4s[n].Eval(xi) for xi in x] for x in xs[n]], axis=1) for n in xs.keys()
    }
  else:
    xs = {
      n:np.array([h.GetBinCenter(i) for i in range(1, h.GetNbinsX()+1)]) for n,h in hs.items()
    }

    g4_vals = {
      n:np.array([g4s[n].Eval(x) for x in xs[n]]) for n in xs.keys()
    }

  h_vals = np.zeros(total)
  a = 0
  for h in hs.values():
    for i in range(h.GetNbinsX()):
      h_vals[a] = h.GetBinContent(i+1)
      a += 1
  return h_vals, g4_vals 

def compare(args):
  #gs = get_g4_xsecs(args)

  a = 0
  fIn = RT.TFile.Open(args.i)
  for n,t in [('abs','Abs'), ('cex','Cex'), ('other','OtherInel')]:
    h = fIn.Get(f'PostFit{t}XSec')
    h.SetDirectory(0)
    xs = np.array([h.GetBinCenter(i) for i in range(1, h.GetNbinsX()+1)])
    ys = np.array([h.GetBinContent(i) for i in range(1, h.GetNbinsX()+1)])
    errs = np.array([h.GetBinError(i) for i in range(1, h.GetNbinsX()+1)])
    gr = RT.TGraphErrors(len(xs), xs, ys, np.zeros(len(xs)), errs)
    c = RT.TCanvas(f'c{n}')
    #h.Draw()
    g4 = fIn.Get(f'{n}_g4')
    ys = np.array(g4.GetY())
    #xs = np.array(g.GetX())
    #g = RT.TGraph(len(xs), xs, ys)
    if n == 'abs': themax = max(ys)
    print(themax)
    g4.SetMaximum(themax*1.1)
    g4.SetLineColor(RT.kRed)
    g4.Draw('AC')
    gr.SetMarkerStyle(20)
    gr.SetMarkerColor(RT.kBlack)
    gr.Draw('same pe1')
    c.SaveAs(f'g4rw_fluc_compare_{n}.pdf')
    a += 1
  fIn.Close()

def get_results(args):
  results_file = RT.TFile.Open(args.results)
  name = {
    'Abs':'abs',
    'Cex':'cex',
    'OtherInel':'other'
  }

  hs = {
    t:results_file.Get(f'PostFitXSec/PostFit{n}XSec') for n,t in name.items() 
  }
  #hs = {
  #  t:results_file.Get(n) for t,n in xsec_strs.items() 
  #}
  print(hs)
  for h in hs.values(): h.SetDirectory(0)
  results_file.Close()
  print('Got results')
  return hs

def make_error_hists(hs, name):
  error_hists = []
  for h in hs.values():
    hErr = h.Clone(f'{h.GetName()}_{name}')
    hErr.Reset()
    error_hists.append(hErr)
  return error_hists

def set_error(hs, cov, content=False):
  a = 0
  for h in hs:
    for i in range(1, h.GetNbinsX()+1):
      if content:
        h.SetBinContent(i, np.sqrt(cov.GetBinContent(a+i, a+i)))
      else:
        h.SetBinError(i, np.sqrt(cov.GetBinContent(a+i, a+i)))
    a += h.GetNbinsX()

def draw_error_conts(total_errs, stat_errs, syst_errs, names, colors,
                     sce_errs=None, frac=False):
  n_hists = len(stat_errs)

  for i in range(n_hists):
    c_errs = RT.TCanvas(f'c_errs_{i}{"_frac" if frac else ""}')
    c_errs.SetTicks()

    for j in range(1, total_errs[i].GetNbinsX()+1):
      if frac:
        err = total_errs[i].GetBinError(j)/total_errs[i].GetBinContent(j)
      else:
        err = total_errs[i].GetBinError(j)
      total_errs[i].SetBinContent(j, err)
      total_errs[i].SetBinError(j, 0.)
    total_errs[i].SetMinimum(0.)
    total_errs[i].SetLineColor(RT.kBlack)
    total_errs[i].SetLineWidth(2)
    total_errs[i].SetMinimum(0.)
    total_errs[i].SetTitle(';Kinetic Energy (MeV);Fractional Error Contribution')
    total_errs[i].Draw('hist')


    stat_errs[i].SetLineColor(RT.kBlack)
    stat_errs[i].SetLineStyle(2)
    stat_errs[i].SetLineWidth(2)
    stat_errs[i].SetMinimum(0.)
    stat_errs[i].Draw('hist same')

    if i == 0:
      leg = RT.TLegend()
      leg.SetFillStyle(0)
      leg.SetLineWidth(0)
      leg.AddEntry(total_errs[i], 'Total', 'l')
      leg.AddEntry(stat_errs[i], 'Stats. Only', 'l')

    a = 0
    styles = [2, 9]
    for syst_err_v in syst_errs:
      syst_err_v[i].SetLineColor(colors[a % len(colors)])
      syst_err_v[i].SetLineWidth(2)
      #if a >= len(colors):
      #  syst_err_v[i].SetLineStyle(2)
      istyle = int(a/len(colors))
      if istyle > 0:
        syst_err_v[i].SetLineStyle(styles[istyle-1])
      syst_err_v[i].Draw('same hist')
      if i == 0:
        leg.AddEntry(syst_err_v[i], names[a], 'l')
      a += 1

    if sce_errs is not None:
      #sce_errs[i].SetLineColor(colors[len(syst_err_v[i])])
      sce_errs[i].SetLineStyle(2)
      sce_errs[i].SetLineWidth(2)
      sce_errs[i].Draw('same hist')
      if i == 0:
        leg.AddEntry(sce_errs[i], 'SCE', 'l')


    if i == 0:
      leg.Draw()
      leg.SetNColumns(3)
    c_errs.Write()

def make_fractional(hs, errs, nodict=False):
  print('Here')
  fracs = []
  for h,e in zip(hs.values() if not nodict else hs, errs):
    print(type(h), type(e))
    fracs.append(e.Clone(e.GetName() + '_frac'))
    fracs[-1].Divide(h)
  return fracs

def errors(args):
  hs = get_results(args)

  with open(args.i, 'r') as fin:
    config = yaml.safe_load(fin)

  syst_files = [sf[1] for sf in config['syst_files']]
  syst_labels = [sf[0] for sf in config['syst_files']]
  syst_names = [sf[2] for sf in config['syst_files']]
  syst_covs = [sf[3] for sf in config['syst_files'] if len(sf) > 3]
  colors = config['colors']

  sce_file = config['sce_file']
  print('Sce file:', sce_file)
  sce_cov = None
  sce_errs = None
  if sce_file:
    sce_root_file = RT.TFile.Open(sce_file)
    sce_cov = sce_root_file.Get('ave_cov')
    sce_cov.SetDirectory(0)
    sce_root_file.Close()
    sce_errs = make_error_hists(hs, 'sce')
    set_error(sce_errs, sce_cov, content=True)


  print(syst_labels)

  #syst_files = args.syst_files
  if len(syst_files) == 0:
    return

  #fin = RT.TFile.Open(args.stat_file)
  print('stat_file', config['stat_file'])
  fin = RT.TFile.Open(config['stat_file'])
  stat_cov = fin.Get('xsec_cov')
  stat_cov.SetDirectory(0)
  fin.Close()

  covs = []
  for i,f in enumerate(syst_files):
    print(f)
    fin = RT.TFile.Open(f)
    if len(syst_covs) > 0:
      n = syst_covs[i] 
    else: n = 'xsec_cov'
    cov = fin.Get(n)
    cov.SetDirectory(0)
    covs.append(cov)
    fin.Close()
  print(covs)

  fout = RT.TFile(args.o, 'recreate')
  total_cov = stat_cov.Clone('hTotalCov')
  total_cov.SetTitle(';Cross Section Bin;Cross Section Bin')
  for c in covs:
    total_cov.Add(c)
  if sce_cov is not None:
    total_cov.Add(sce_cov)

  total_cov.Write()
  total_cov.Write('xsec_cov')

  total_corr = total_cov.Clone('hTotalCorr')
  total_corr.SetMinimum(-1)
  total_corr.SetMaximum(1)
  total_corr.SetTitleSize(.06, "XY")
  for i in range(1, total_corr.GetNbinsX()+1):
    content_i = total_cov.GetBinContent(i,i)
    for j in range(1, total_cov.GetNbinsX()+1):
      content_j = total_cov.GetBinContent(j,j)
      content_ij = total_cov.GetBinContent(i,j)
      total_corr.SetBinContent(i, j, content_ij/np.sqrt(content_i*content_j))
  total_corr.Write()
  total_corr.Write('xsec_corr')


  set_error(hs.values(), total_cov)
  for n,h in hs.items():
    h.Write()
    xs = np.array([h.GetBinCenter(i) for i in range(1, h.GetNbinsX()+1)])
    ys = np.array([h.GetBinContent(i) for i in range(1, h.GetNbinsX()+1)])
    errs = np.array([h.GetBinError(i) for i in range(1, h.GetNbinsX()+1)])
    gr = RT.TGraphAsymmErrors(len(xs), xs, ys, np.zeros(len(xs)), np.zeros(len(xs)), errs, errs)
    gr.Write(f'{n}_xsec')


  stat_errs = make_error_hists(hs, 'stat')
  set_error(stat_errs, stat_cov, content=True)
  for h in stat_errs: h.Write()

  fracs_stat = make_fractional(hs, stat_errs)
  for h in fracs_stat: h.Write()

  i = 0
  syst_errs = []
  syst_fracs = []
  for cov in covs:
    errs = make_error_hists(hs, syst_names[i])
    i += 1
    set_error(errs, cov, content=True)
    syst_errs.append(errs)
    for h in errs: h.Write()

    fracs = make_fractional(hs, errs)
    syst_fracs.append(fracs)
    for h in fracs: h.Write()


  hs = [h for h in hs.values()]
  frac_hs = [h.Clone() for h in hs]
  fracs_sce = None
  if sce_errs is not None:
    for sce_err in sce_errs:
      sce_err.Write()
    fracs_sce = make_fractional(hs, sce_errs, nodict=True)
    for h in fracs_sce: h.Write()

  draw_error_conts(hs, stat_errs, syst_errs, syst_labels, colors, sce_errs=sce_errs)
  draw_error_conts(frac_hs, fracs_stat, syst_fracs, syst_labels, colors,
                    sce_errs=fracs_sce, frac=True)

  fG4 = RT.TFile.Open(config['g4_file'])
  g4s = get_g4_xsecs_config(fG4)
  fout.cd()
  for n, g4 in g4s.items(): g4.Write(f'{n}_g4')

  #colors = {'abs':RT.kBlue, 'cex':RT.kRed, 'other':RT.kBlack}

  fG4.Close()

  fout.Close()

def plot_chi2(chi2s, fit_ndf, exp_ndf, xlabel='Cross Section $\chi^2$'):
  bins = np.arange(0, 100, 1) - .5
  plt.hist(chi2s, bins=bins, density=True)
  plt.plot(statschi2.pdf(bins+.5, fit_ndf), label=f'NDF={fit_ndf:.2f}')
  plt.plot(statschi2.pdf(bins+.5, exp_ndf), label=f'NDF={exp_ndf}',
           linestyle='--', color='orange')
  plt.xlabel(xlabel)
  plt.ylabel('Fraction of Fits')
  plt.xticks()
  plt.yticks()
  plt.legend()
  plt.show()

def process_chi2(args):
  lines = open_list(args.i)
  chi2s = get_all_chi2s(args, lines)
  bins = np.arange(0, 100, 1) - .5
  plt.hist(np.sum(chi2s, axis=1), bins=bins, density=True, label='Fits')
  plt.plot(statschi2.pdf(bins+.5, args.ndf), label=f'NDF={args.ndf}',
           linestyle='--', color='orange')
  plt.xlabel('Minimum -2LLH')
  plt.ylabel('Number of Fits [a.u.]')
  plt.legend()
  plt.show()

def open_processed(args):
  f = RT.TFile.Open(args.i)
  hcov = f.Get('xsec_cov')
  hcorr = f.Get('xsec_corr')
  hcorr.SetDirectory(0)

  nbins = hcov.GetNbinsX()
  cov = np.zeros((nbins, nbins))
  for i in range(nbins):
    for j in range(nbins):
      cov[i, j] = hcov.GetBinContent(i+1, j+1)
  f.Close()
  return cov, hcorr

def process_results(args):
  RT.gROOT.LoadMacro("~/protoDUNEStyle.C")
  RT.gROOT.SetBatch();
  RT.gStyle.SetOptStat(00000)
  RT.gStyle.SetErrorX(1.e-4)
  RT.gStyle.SetTitleAlign(33)
  RT.gStyle.SetTitleX(.5)
  g4s = get_g4_xsecs(args)
  h_results = get_results(args)
  cov, hcorr = open_processed(args)
  fout = RT.TFile(args.o, 'recreate')
  plot_results(h_results, g4s, cov)

  c = RT.TCanvas('cCorr')
  c.SetTicks()
  hcorr.SetTitle(';Cross Section Bin;Cross Section Bin')
  hcorr.GetXaxis().CenterTitle()
  hcorr.SetMinimum(-1.)
  hcorr.GetYaxis().CenterTitle()
  hcorr.GetXaxis().SetNdivisions(13)
  hcorr.GetYaxis().SetNdivisions(13)
  hcorr.Draw('colz')
  tt.DrawLatex(0.10,0.94,"#bf{DUNE:ProtoDUNE-SP}");
  c.Write()

  fout.Close()


def draw_errs(args):
  RT.gROOT.LoadMacro("~/protoDUNEStyle.C")
  RT.gROOT.SetBatch();
  RT.gStyle.SetOptStat(00000)
  RT.gStyle.SetErrorX(1.e-4)
  RT.gStyle.SetTitleAlign(33)
  RT.gStyle.SetTitleX(.5)

  cov, hcorr = open_processed(args)
  c = RT.TCanvas('cCorr')
  c.SetTicks()
  hcorr.SetTitle(';Cross Section Bin;Cross Section Bin')
  hcorr.GetXaxis().CenterTitle()
  hcorr.SetMinimum(-1.)
  hcorr.GetYaxis().CenterTitle()
  hcorr.GetXaxis().SetNdivisions(13)
  hcorr.GetYaxis().SetNdivisions(13)
  hcorr.Draw('colz')
  tt.DrawLatex(0.10,0.94,"#bf{DUNE:ProtoDUNE-SP}");
  fout = RT.TFile(args.o, 'recreate')
  c.Write()

  c = RT.TCanvas('cVar')
  nbins = hcorr.GetNbinsX()
  hVar = RT.TH1D('hvar', ';Cross Section Bin;Variance [mb^{2}]', nbins, 0, nbins)
  for i in range(1, nbins+1):
    hVar.SetBinContent(i, cov[i-1][i-1])
  hVar.Draw()
  hVar.GetYaxis().CenterTitle()
  hVar.GetXaxis().CenterTitle()
  tt.DrawLatex(0.10,0.94,"#bf{DUNE:ProtoDUNE-SP}");
  c.Write()
  fout.Close()
  
def comp_errs(args):
  ferrs = RT.TFile.Open(args.i)
  fpulls = RT.TFile.Open(args.results)

  names = ['Abs', 'Cex',  'OtherInel']
  base = 0
  for n in names:
    h_low = ferrs.Get(f'PostFit{n}XSec_pilow')
    h_high = ferrs.Get(f'PostFit{n}XSec_pihigh')
    h = ferrs.Get(f'PostFit{n}XSec')

    print(n)
    for i in range(1, 4):
      h_pull = fpulls.Get(f'h{i - 1 + base}')
      err = np.sqrt((h_low.GetBinContent(i))**2 + (h_high.GetBinContent(i))**2)
      print(f'{i} -- Pull {h_pull.GetMean():.4f} Err {err/h.GetBinContent(i):.4f}')
      
    base += 3
      

def comp_pulls(args):
  filenames = args.i.split(',')

  nfiles = len(filenames)
  nxsecs = 9
  means = np.zeros((nxsecs, nfiles))
  errs = np.zeros((nxsecs, nfiles))

  j = 0
  for n in filenames:
    #print(n)
    f = RT.TFile.Open(n)
    #print(f.Get('h0').GetMean(), f.Get('h0').GetStdDev())
    for i in range(nxsecs):
      means[i,j] = f.Get(f'h{i}').GetMean()
      errs[i,j] = f.Get(f'h{i}').GetStdDev()
    f.Close()
    j += 1

  titles = [
    'Absorption (500-600) MeV',
    'Absorption (600-700) MeV',
    'Absorption (700-800) MeV',
    'Ch. Exch. (500-600) MeV',
    'Ch. Exch. (600-700) MeV',
    'Ch. Exch. (700-800) MeV',
    'Other (500-600) MeV',
    'Other (600-700) MeV',
    'Other (700-800) MeV',
  ]

  from matplotlib.ticker import MaxNLocator
  if args.save is not None:
    plt.ion()
  for i in range(nxsecs):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.errorbar(1 + np.arange(nfiles), means[i], errs[i], label=titles[i])
    plt.plot(1 + np.arange(nfiles), np.zeros(nfiles))
    plt.ylim(-1,1)
    #plt.xticks(['Base', *[f'Iter. {n}' for n in range(nfiles)]])
    plt.xlabel('Fit Iterations')
    plt.ylabel('r')
    plt.legend()
    if args.save is not None:
      plt.savefig(f'r_iters_{args.save}_xsec_{i}.png')
      plt.savefig(f'r_iters_{args.save}_xsec_{i}.pdf')
      plt.close()
    else:
      plt.show()


def extra_unc(args):
  f = RT.TFile.Open(args.i) 
  xsecs = []
  for name in ['abs', 'cex', 'other']:
    g = f.Get(f'{name}_xsec')
    for i in range(g.GetN()): xsecs.append(g.GetY()[i])
  n = len(xsecs)
  cov = RT.TH2D('xsec_cov', '', n, 0, n, n, 0, n)
  for i in range(n): cov.SetBinContent(i+1, i+1, (.05*xsecs[i])**2)

  fout = RT.TFile(args.o, 'recreate')
  cov.Write()
  fout.Close()

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', required=True)
  parser.add_argument('-o', default='g4rw_fluc_out.root')
  parser.add_argument('--nocheck', action='store_true', help='Prevent the good-hesse check')
  parser.add_argument('--routine', type=str, default='process',
                      choices=['process', 'compare', 'errors', 'chi2',
   'results', 'test', 'variation', 'draw_errs',
   'comp_errs', 'comp_pulls', 'extra_unc',
  ],
                      help='Options: process, compare, errors, chi2')
  parser.add_argument('--g4', type=str, default='/exp/dune/data/users/calcuttj/old_data2/PiAnalysis_G4Prediction/thresh_abscex_xsecs.root')
  parser.add_argument('--scales', type=float, nargs=3, default=[1., 1., 1.])
  parser.add_argument('--fit', action='store_true', help='Use with process routine')
  parser.add_argument('--results', type=str, help='Use with process routine')
  parser.add_argument('--exclude', action='store_true')
  parser.add_argument('--fake', action='store_true', help='Use with process routine')
  parser.add_argument('--nofake', action='store_true', help='Use with process routine')
  parser.add_argument('--fake_diff', action='store_true', help='Use with process routine')
  parser.add_argument('--ave_g4', action='store_true')
  parser.add_argument('--throws', action='store_true')

  #parser.add_argument('--syst_files', type=str, nargs='+',
  #                    help='Use with errors routine')
  #parser.add_argument('--stat_file', type=str,
  #                    help='Use with errors routine')
  parser.add_argument('--ints', help='Use with prcocess routine', action='store_true')
  parser.add_argument('--ndf', type=int, default=18)
  parser.add_argument('--save', default=None, type=str)
  args = parser.parse_args()

  
  routines = {'process':process,
              'compare':compare,
              'errors':errors,
              'chi2':process_chi2,
              'results':process_results,
              'test':test_files,
              #'variation':variation,
              'draw_errs':draw_errs,
              'comp_errs':comp_errs,
              'comp_pulls':comp_pulls,
              'extra_unc':extra_unc,
  }

  if args.routine in ['process', 'compare', 'errors', 'chi2']:
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.stats import chi2 as statschi2
    from scipy.optimize import minimize
    import yaml
  routines[args.routine](args)
  #if args.routine =='process':
  #  from scipy import stats
  #  from scipy.optimize import minimize
  #  from scipy.stats import chi2 as statschi2
  #  from scipy.optimize import minimize
  #  import yaml
  #  process(args)
  #elif args.routine == 'compare':
  #  from scipy import stats
  #  from scipy.optimize import minimize
  #  from scipy.stats import chi2 as statschi2
  #  from scipy.optimize import minimize
  #  import yaml
  #  compare(args)
  #elif args.routine == 'errors':
  #  from scipy import stats
  #  from scipy.optimize import minimize
  #  from scipy.stats import chi2 as statschi2
  #  from scipy.optimize import minimize
  #  import yaml
  #  errors(args) 
  #elif args.routine == 'chi2':
  #  from scipy import stats
  #  from scipy.optimize import minimize
  #  from scipy.stats import chi2 as statschi2
  #  from scipy.optimize import minimize
  #  import yaml
  #  process_chi2(args)
  #elif args.routine == 'results':
  #  process_results(args)
  #elif args.routine == 'test':
  #  test_files(args)
  #elif args.routine == 'variation':
  #  variation(args)
  #elif args.routine == 'draw_errs':
  #  draw_errs(args)
  #elif args.routine == 'comp_errs':
  #  comp_errs(args)
  #elif args.routine == 'comp_pulls':
  #  comp_pulls(args)
