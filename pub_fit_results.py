import ROOT as RT
import numpy as np
import math
from array import array
import sys
from argparse import ArgumentParser as ap

def get_new_xsec(f, name):
  
  if name == 'abs': title = 'Abs'
  elif name == 'cex': title = 'Cex'
  elif name == 'other': title = 'OtherInel'

  hist = f.Get(f"PostFitXSec/PostFit{title}XSec")
  xs = []
  ys = []
  eyhs = []
  eyls = []
  for i in range(1, hist.GetNbinsX()+1):
    xs.append(hist.GetBinCenter(i))
    ys.append(hist.GetBinContent(i))
    eyhs.append(0.)
    eyls.append(0.)
  
  n = len(xs)
  print(n)
  new_gr = RT.TGraphAsymmErrors(
    n,
    np.array(xs),
    np.array(ys),
    np.array([0.]*n),
    np.array([0.]*n),
    np.array(eyls),
    np.array(eyhs)
  )
  return new_gr

def combine(hists, title,
            labels=["Failed Beam Cuts",
                    "No Beam Track",
                    "Michel Vertex Cut"]):

  n = len(labels)

  combined = RT.TH1D(title, "", n, 0, n)
  datas = [h for h in hists[-n:]]
  print(datas)
  #data1 = hists[-3]
  #data2 = hists[-2]
  #data3 = hists[-1]
  #combined.SetBinContent(1, data1.GetBinContent(1))
  #combined.SetBinContent(2, data2.GetBinContent(1))
  #combined.SetBinContent(3, data3.GetBinContent(1))
  for i,d in enumerate(datas):
    combined.SetBinContent(i+1, d.GetBinContent(1))
    combined.GetXaxis().SetBinLabel(i+1, labels[i])

  #combined.GetXaxis().SetBinLabel(1, "Failed Beam Cuts")
  #combined.GetXaxis().SetBinLabel(2, "No Beam Track")
  #combined.GetXaxis().SetBinLabel(3, "Michel Vertex Cut")
  combined.GetXaxis().SetLabelSize(.05)
  combined.SetMinimum(0.)
  return combined
  
def rebin(hists, title):
  results = []
  for i in range(0, 4):
    h = hists[i]
    bins = []
    for j in range(1, h.GetNbinsX()+1):
      label = h.GetXaxis().GetBinLabel(j)
      print('label', label)
      if j == 1:     
        bins.append(float(label.split(' ')[0]))
      bins.append(float(label.split(' ')[-1]))
    # if i == 3: # For APA2 fix this 
    #   bins = [215., 580.]
    print(bins)
    rebinned = RT.TH1D(hists[i].GetName() + title + "_rebinned", hists[i].GetTitle(),
                       len(bins) -1, array('d', bins)) 
    for j in range(1, h.GetNbinsX()+1):
      rebinned.SetBinContent(j, h.GetBinContent(j))
    results.append(rebinned)
  return results 




def fix_params_plot(args):
  #RT.gROOT.LoadMacro("~/protoDUNEStyle.C")
  #RT.gROOT.SetBatch();
  #RT.gStyle.SetOptStat(00000)
  #RT.gStyle.SetTitleAlign(33)
  #RT.gStyle.SetTitleX(.915)
  #RT.gStyle.SetTitleY(.95)
  #RT.gROOT.ForceStyle()
  RT.gROOT.LoadMacro("~/style.C")
  RT.gROOT.ProcessLine('style()')
  RT.gStyle.SetErrorX(0.5)

  #if not args.noplotstyle:
  #  import dunestyle.root as dunestyle

  f = RT.TFile.Open(args.i)

  h = f.Get('postFitPars')
  hpre = f.Get('preFitPars')
  hnew = RT.TH1D('hnew', '', args.npars, 0, args.npars)
  hnew_pre = RT.TH1D('hnew_pre', '', args.npars, 0, args.npars)
  for i in range(1, args.npars+1):
    hnew.SetBinContent(i, h.GetBinContent(i))
    hnew.SetBinError(i, h.GetBinError(i))
    print('error:', h.GetBinError(i))

    hnew_pre.SetBinContent(i, hpre.GetBinContent(i))
    hnew_pre.SetBinError(i, hpre.GetBinError(i))

  minval = 0.
  maxval = 2.
  for i in range(1, args.npars+1):
    a = hnew.GetBinContent(i) + 2.*hnew.GetBinError(i)
    if a > maxval: maxval = a

    b = hnew.GetBinContent(i) - 2.*hnew.GetBinError(i)
    if b < minval: minval = b

  print('Setting', minval, maxval)
  hnew.SetMinimum(minval)
  hnew.SetMaximum(maxval)
  hnew.GetXaxis().SetTitleOffset(.8);
  hnew.GetYaxis().SetTitleOffset(.8);
  hnew.SetTitleSize(.05, "XY");
  hnew.SetMarkerColor(RT.kRed);
  hnew.SetMarkerStyle(20);
  hnew.GetXaxis().SetRangeUser(0, args.npars)
  hnew.SetFillStyle(3001)
  hnew.SetFillColor(RT.kRed)
  hnew.SetLineWidth(0)
  hnew.SetTitle(';Parameter;Parameter Value')

  c = RT.TCanvas('cParameters')
  c.SetTicks()
  hnew.Draw('pe2')
  hnew_pre.SetLineColor(RT.kBlue)
  hnew_pre.SetMarkerColor(RT.kBlue)
  hnew_pre.SetMarkerStyle(20)
  hnew_pre.Draw('p same')
  #fOut.cd()
  l = RT.TLine(0., 1., args.npars, 1.)
  #l.SetLineWidth(2)
  l.Draw()
  hnew.Draw('pe2 same')
  leg = RT.TLegend()
  leg.AddEntry(hnew_pre, 'Pre-Fit', 'lp')
  leg.AddEntry(hnew, 'Post-Fit #pm1#sigma', 'lpf')
  leg.SetFillStyle(0)
  leg.SetLineWidth(0)
  leg.Draw()
  c.RedrawAxis()

  c2D = f.Get('cCorr') 
  c2D.SetTicks()
  c2D.Draw()

  prims = c2D.GetListOfPrimitives()
  for i in range(prims.GetSize()):
    if 'hist' in prims.At(i).GetName().lower(): break
  prims.At(i).GetXaxis().SetRangeUser(0, args.npars)
  prims.At(i).GetYaxis().SetRangeUser(0, args.npars)
  c2D.Update()
  c2D.RedrawAxis()



  if args.save:
    c.SaveAs('pars.pdf')
    c2D.SaveAs('corr.pdf')
  #else:
  #  fOut = RT.TFile(args.o, 'recreate')
  #  c.Write()
  #  fOut.Close()

  f.Close()

def normal(args):

  
  RT.gROOT.LoadMacro("~/protoDUNEStyle.C")
  RT.gROOT.SetStyle("protoDUNEStyle")
  RT.gROOT.ForceStyle()
  RT.gStyle.SetTitleX(0.5)
  RT.gStyle.SetTitleAlign(22)
  RT.gStyle.SetTitleY(0.87)
  RT.gStyle.SetTitleW(0.80) # or .85
  RT.gStyle.SetOptFit(111)

  # RT.gROOT.SetBatch();
  # RT.gStyle.SetOptStat(00000)
  # RT.gStyle.SetErrorX(1.e-4)
  # RT.gStyle.SetTitleAlign(33)
  # RT.gStyle.SetTitleX(.915)
  # RT.gStyle.SetTitleY(.95)
  # RT.gROOT.ForceStyle()
  
  prefit_names = [
    "NominalAbsTotal", "NominalCexTotal",
    "NominalRejectedIntTotal", "NominalAPA2Total",
  ]
  
  data_names = [
    "Data/Data_selected_Abs_hist", "Data/Data_selected_Cex_hist",
    "Data/Data_selected_RejectedInt_hist", "Data/Data_selected_APA2_hist",
  ]

  prefit_hists = []
  
  total_prefit = 0.
  fPrefit = RT.TFile.Open(args.prefit)
  for n in prefit_names:
    prefit_hists.append(fPrefit.Get(n))
    prefit_hists[-1].SetDirectory(0)
    total_prefit += prefit_hists[-1].Integral()

  data_hists = []
  total_data = 0.
  for n in data_names:
    data_hists.append(fPrefit.Get(n))
    data_hists[-1].SetDirectory(0)
    total_data += data_hists[-1].Integral()
  
  fPrefit.Close()
  
  fPostfit = RT.TFile.Open(args.postfit)
  
  postfit_names = [
    "PostFitAbsTotal", "PostFitCexTotal",
    "PostFitRejectedIntTotal", "PostFitAPA2Total",
  ]

  postfit_hists = []
  for n in postfit_names:
    postfit_hists.append(fPostfit.Get(n))
    postfit_hists[-1].SetDirectory(0)
  fPostfit.Close()

  new_data_hists = []
  for h, hMC in zip(data_hists, prefit_hists):
    # h.Scale(total_prefit/total_data)
    new_h = hMC.Clone()
    new_h.Reset()
    for i in range(1, h.GetNbinsX()+1):
      new_h.SetBinContent(i, h.GetBinContent(i))
    new_data_hists.append(new_h)
  
  names = [
    "Absorption Cand.", "Charge Exchange Cand.",
    "Other Interaction Cand.", "Past Fiducial Volume",
    "Absorption Cand.", "Charge Exchange Cand.",
    "Other Interaction Cand.", "Past Fiducial Volume",
  ]
  
  short_names = [
    "Abs", "Cex",
    "RejectedInt", "APA2",
    # "FailedBeamCuts", "NoBeamTrack"
  ]
  
  short_names = [
    "Abs", "Cex",
    "RejectedInt", "APA2",
    # "FailedBeamCuts", "NoBeamTrack",
    # "MichelCut", "BadEvents",
    "AbsRebin", "CexRebin",
    "RejectedIntRebin", "APA2Rebin",
  ]
  
  XTitles = [
    "Reconstructed KE [MeV]",
    "Reconstructed KE [MeV]",
    "Reconstructed KE [MeV]",
    "Reconstructed End Z [cm]",
    # "",
    # "",
    # "", "",
    "Reconstructed KE [MeV]",
    "Reconstructed KE [MeV]",
    "Reconstructed KE [MeV]",
    "Reconstructed End Z [cm]",
  ]
  
  #fOut = RT.TFile(sys.argv[2], "RECREATE");
  fOut = RT.TFile(args.o, "RECREATE");
  
  # labels=[
  #   "Failed Beam Cuts",
  #   "No Beam Track",
  # ]
  # if not args.no_michel: labels.append("Michel Vertex Cut")

  
  new_data_hists += rebin(new_data_hists, 'data')
  prefit_hists += rebin(prefit_hists, 'prefit')
  postfit_hists += rebin(postfit_hists, 'postfit')
  tt = RT.TLatex();
  tt.SetNDC();
  for i in range(0, len(prefit_hists)):
    data = new_data_hists[i]
    data.SetMarkerStyle(20);
    data.SetMarkerColor(RT.kBlack);
    #data.SetMarkerSize(.5)
    data.SetLineColor(RT.kBlack);
    data.SetTitle(names[i] + ";" + XTitles[i] + ";Events per bin [x10^{3}]")
    data.GetYaxis().SetMaxDigits(3)
    #data.SetTitleSize(.05, "XY")
    data.GetXaxis().CenterTitle()
    data.GetYaxis().CenterTitle()
    data.Scale(1.e-3)
    #data.GetXaxis().SetTitleOffset(.8)
    data.GetYaxis().SetTitleOffset(.95)
    data.SetMinimum(0.)
    #if (i == len(prefit_hists)-1):
    #  data.SetTitle(names[i])
    #else:
    #  data.SetTitle(names[i] + ";" + XTitles[i])
  
    prefit = prefit_hists[i]
    prefit.SetFillColor(0);
    prefit.SetLineStyle(2);
    prefit.SetLineColor(RT.kBlue);
    prefit.SetTitle(names[i] + ";" + XTitles[i] + ";Events per bin [x10^{3}]")
    prefit.GetXaxis().CenterTitle()
    prefit.GetYaxis().CenterTitle()
    prefit.Scale(1.e-3)
    #prefit.SetTitleSize(.05, "XY")
    #prefit.GetXaxis().SetTitleOffset(.8)
    prefit.GetYaxis().SetTitleOffset(.95)
    prefit.GetYaxis().SetMaxDigits(3)
    prefit.SetMinimum(0.)
    #if (i == len(prefit_hists)-1):
    #  prefit.SetTitle(names[i])
    #else:
    #  prefit.SetTitle(names[i] + ";Reconstructed KE (MeV)")
  
    postfit = postfit_hists[i]
    postfit.SetFillColor(0);
    postfit.SetLineColor(RT.kRed)
    # postfit.SetLineStyle(2)
    postfit.SetTitle(names[i] + ";" + XTitles[i] + ";Events per bin [x10^{3}]")
    postfit.GetYaxis().SetMaxDigits(3)
    postfit.GetXaxis().CenterTitle()
    postfit.GetYaxis().CenterTitle()
    #postfit.SetTitleSize(.05, "XY")
    #postfit.GetXaxis().SetTitleOffset(.8)
    postfit.GetYaxis().SetTitleOffset(.95)
    postfit.SetMinimum(0.)
    postfit.Scale(1.e-3)
    #if (i == len(prefit_hists)-1):
    #  postfit.SetTitle(names[i])
    #else:
    #  postfit.SetTitle(names[i] + ";Reconstructed KE (MeV)")
  
    # if i == 3:
    #   data.GetXaxis().SetLabelOffset(.01)
    #   prefit.GetXaxis().SetLabelOffset(.01)
    #   postfit.GetXaxis().SetLabelOffset(.01)
    #   data.GetXaxis().SetLabelSize(.07)
    #   prefit.GetXaxis().SetLabelSize(.07)
    #   postfit.GetXaxis().SetLabelSize(.07)
  
  
    c = RT.TCanvas("c" + short_names[i], "c" + short_names[i])
    c.SetTicks()
  
    post_bins = [postfit.GetBinContent(i) for i in range(1, postfit.GetNbinsX()+1)]
    pre_bins = [prefit.GetBinContent(i) for i in range(1, prefit.GetNbinsX()+1)]
    data_bins = [data.GetBinContent(i) + data.GetBinError(i) for i in range(1, data.GetNbinsX()+1)]
    # if 'Michel' in short_names[i] or 'APA2' in short_names[i]:
    #   postfit.GetXaxis().SetBinLabel(1, '')
    #   prefit.GetXaxis().SetBinLabel(1, '')
    #   data.GetXaxis().SetBinLabel(1, '')
      
    max_post = max(post_bins)
    max_pre = max(pre_bins)
    max_data = max(data_bins)
    
    the_max = max([max_post, max_pre, max_data])
  
    # if (max_post > max_pre and max_post > max_data):
    postfit.SetMaximum(2.*the_max)
    postfit.SetMinimum(0.)
    postfit.Draw("hist ][")
    prefit.Draw("hist same ][")
    data.Draw("same e ][")
    # elif (max_pre > max_post and max_pre > max_data):
    #   prefit.Draw("hist")
    #   postfit.Draw("hist same")
    #   prefit.Draw("hist same")
    #   data.Draw("same e")
    # elif (max_data > max_pre and max_data > max_post):
    #   data.Draw("e")
    #   postfit.Draw("hist same")
    #   prefit.Draw("hist same")
    #   data.Draw("same e")
  
    tt.DrawLatex(0.10,0.94,"#bf{DUNE:ProtoDUNE-SP}");
  
  
    if (i == 0):
      l = RT.TLegend(.2, .65, .8, .83)
      l.AddEntry(prefit, "Pre-Fit", "l")
      l.AddEntry(postfit, "Post-Fit", "l")      
      l.AddEntry(data, "Data", "pe")
      l.SetNColumns(2)
      h = RT.TH1D("dummy", "", 1, 0, 1)
      h.SetLineColor(0)
      l.Write("leg")
      l.SetFillStyle(0)
      l.SetLineWidth(0)
      #l.AddEntry(h, "Nominal " + nominal_label, "")
      #l.AddEntry(h, "Post-Fit " + postfit_label, "")
    if i in [0, 4]:
      l.Draw()
    RT.gPad.RedrawAxis()
    c.Write()
    c.SaveAs('fits_' + short_names[i] + '.pdf')
    c.SaveAs('fits_' + short_names[i] + '.png')
    #if args.save:
    #  c.SaveAs("c" + short_names[i] + ".pdf")
  
  
    
  fOut.Close();


def save(args):
  RT.gROOT.LoadMacro("~/style.C")
  RT.gROOT.ProcessLine('style()')
  f = RT.TFile.Open(args.i)
  cs = [
    'cAbsRebin', 'cCexRebin', 'cRejectedIntRebin',
    'cAPA2', 'cBadEvents', 'cMichelCut',
  ]
  if args.no_michel: cs.pop(cs.index('cMichelCut'))
  for n in cs:
    c = f.Get(n)
    c.Draw()
    c.SaveAs(n + '.pdf') 
  f.Close()

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-o', type=str, default='')
  parser.add_argument('--prefit', type=str, required=True)
  parser.add_argument('--postfit', type=str, required=True)


  parser.add_argument('--pretune', type=str, default=None)
  parser.add_argument('--add_files', nargs='+', default=[])
  parser.add_argument('--add_covs', nargs='+', default=[])
  parser.add_argument('--fixed', action='store_true')
  parser.add_argument('--nothrows', action='store_true')
  parser.add_argument('--save', action='store_true')
  parser.add_argument('--routine', type=str, default='normal',
                      choices=[
                        'normal', 'pars', 'save'
                      ])
  parser.add_argument('--npars', default=0, type=int)
  parser.add_argument('--noplotstyle', action='store_true')
  args = parser.parse_args()
  
  routines = {
    'normal':normal,
    'pars':fix_params_plot,
    'save':save,
  }
  routines[args.routine](args)
