import ROOT as RT
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import numpy as np
RT.gROOT.SetBatch()
RT.gStyle.SetOptStat(0)
RT.gStyle.SetErrorX(0.)
from argparse import ArgumentParser as ap

colors = [
 602, 433, 837, 419,
 403, 795, 630, 629,
 893, 619, 397, 632
]

def get_vals(t, outname, maxnum=-1):
  ####Efficiency testing

  fOut = RT.TFile(outname.replace('.h5', '.root'), 'recreate')
  tOut = RT.TTree('tree', 'tree')
  x_br = np.array([0.])
  cat_br = np.array([0])
  y_br = np.array([0.])
  cos_br = np.array([0.])
  cos_end_br = np.array([0.])
  beam_x_br = np.array([0.])
  beam_y_br = np.array([0.])
  good_event_br = np.array([0])
  tOut.Branch('good_event', good_event_br, 'good_event/I')
  tOut.Branch('cat', cat_br, 'cat/I')
  tOut.Branch('x', x_br, 'x/D')
  tOut.Branch('y', y_br, 'y/D')
  tOut.Branch('cos', cos_br, 'cos/D')
  tOut.Branch('cos_end', cos_end_br, 'cos_end/D')
  tOut.Branch('beam_x', beam_x_br, 'beam_x/D')
  tOut.Branch('beam_y', beam_y_br, 'beam_y/D')

  with h5.File(outname, 'w') as h5f:
    dsets = dict()
    sets = [
      'xs', 'ys', 'costhetas', 'costhetas_end', 'beam_xs', 'beam_ys',
      'true_cat'
    ]
    #zcuts = np.arange(0., 20., 5.)
    zcuts = [30.]
    for n in sets:
      dsets[n] = h5f.create_dataset(n,
                              (len(zcuts), ),
                              dtype=h5.vlen_dtype(np.dtype('float32')))
    xs = [[] for i in range(len(zcuts))]
    ys = [[] for i in range(len(zcuts))]
    true_cat = [[] for i in range(len(zcuts))]
    costhetas = [[] for i in range(len(zcuts))]
    print(len(xs))
    costhetas_end = [[] for i in range(len(zcuts))]

    beam_xs = [[] for i in range(len(zcuts))]
    beam_ys = [[] for i in range(len(zcuts))]

    for ie, e in enumerate(t):
      if not ie % 1000: print(f'{ie}/{t.GetEntries()}', end='\r')
      #if e.true_beam_PDG not in [211, -13]: continue
      if e.selection_ID == 6: continue
      #if len(e.reco_beam_incidentEnergies) == 0 and e.reco_beam_type != 13:
      #  continue
      if e.reco_beam_calo_startZ < 0.: continue
      #calo_Z = np.array(e.reco_beam_calo_Z)
      #calo_X = np.array(e.reco_beam_calo_X)
      #calo_Y = np.array(e.reco_beam_calo_Y)
      calo_Z = e.reco_beam_calo_Z
      calo_X = e.reco_beam_calo_X
      calo_Y = e.reco_beam_calo_Y

      zl = e.reco_beam_calo_endZ
      xl = e.reco_beam_calo_endX
      yl = e.reco_beam_calo_endY

      beam_dirX = e.beam_inst_dirX
      beam_dirY = e.beam_inst_dirY
      beam_dirZ = e.beam_inst_dirZ



      for i, zcut in enumerate(zcuts):

        in_fv = zl > zcut
        if not in_fv:

          if i == 0:
            x_br[0] = -999.
            y_br[0] = -999.
            cos_br[0] = -999.
            cos_end_br[0] = -999.
            beam_x_br[0] = -999.
            beam_y_br[0] = -999.
            cat_br[0] = -999
            good_event_br[0] = 0
            tOut.Fill()
          continue
        good_event_br[0] = 1
        #print(zcut, np.where(calo_Z >= zcut), np.where(calo_Z >= zcut)[0][0])
        #iz = max(np.where(calo_Z >= zcut)[0][0], 1)
        for iz in range(len(calo_Z)):
          if calo_Z[iz] > zcut: break
        if iz < 1: iz = 1
        z0 = calo_Z[iz-1]
        z1 = calo_Z[iz]
        x0 = calo_X[iz-1]
        x1 = calo_X[iz]
        y0 = calo_Y[iz-1]
        y1 = calo_Y[iz]

        tpc_x = x0 + (zcut - z0)*(x1-x0)/(z1 - z0)

        tpc_y = y0 + (zcut - z0)*(y1-y0)/(z1 - z0)
        xs[i].append(tpc_x)
        ys[i].append(tpc_y)

        r = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        r_end = np.sqrt((xl-tpc_x)**2 + (yl-tpc_y)**2 + (zl-zcut)**2)
        if r_end <= 0.:
          print('Warning', r_end, iz, xl, x0, yl, y0, zl, z0)
          print(calo_X)
          print(calo_Z)


        #print(r_end)
        costhetas_end[i].append(
          ((xl-tpc_x)/r_end*beam_dirX + (yl-tpc_y)/r_end*beam_dirY + (zl-zcut)/r_end*beam_dirZ)
        )
        beam_xs[i].append(e.beam_inst_X + zcut*beam_dirX/beam_dirZ)
        beam_ys[i].append(e.beam_inst_Y + zcut*beam_dirY/beam_dirZ)
        if e.MC:
          true_cat[i].append(e.beam_backtrack)
        if i == 0:
          x_br[0] = tpc_x 
          y_br[0] = tpc_y
          cos_br[0] = -999.
          cos_end_br[0] = costhetas_end[i][-1]
          beam_x_br[0] = beam_xs[i][-1] 
          beam_y_br[0] = beam_ys[i][-1] 
          if e.MC: cat_br[0] = e.beam_backtrack
          tOut.Fill()
         
      if maxnum > 0 and ie > maxnum: break

    #h5f['zs'] = zcuts
    fOut.cd()
    tOut.Write()
    fOut.Close()

    for i in range(len(zcuts)):
      dsets['xs'][0] = xs[i]
      dsets['ys'][0] = ys[i]
      dsets['beam_xs'][0] = beam_xs[i]
      dsets['beam_ys'][0] = beam_ys[i]
      dsets['costhetas'][0] = costhetas[i]
      dsets['costhetas_end'][0] = costhetas_end[i]
      dsets['true_cat'][0] = true_cat[i]


     
def gauss(x, a, x0, sigma):
  return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_stats(args):
  from scipy.optimize import curve_fit
  with h5.File(args.i, 'r') as f:
    dxs = f['xs'][0] - f['beam_xs'][0]
    dys = f['ys'][0] - f['beam_ys'][0]
    drs = np.sqrt(dxs**2 + dys**2)

    dxs = dxs[np.where(abs(dxs) < 30.)]
    dys = dys[np.where(abs(dys) < 30.)]
    drs = drs[np.where(abs(drs) < 30.)]

    mean_x, sigma_x = np.mean(dxs), np.std(dxs)
    mean_y, sigma_y = np.mean(dys), np.std(dys)
    mean_r, sigma_r = np.mean(drs), np.std(drs)

    print('X:', mean_x, sigma_x)
    print('Y:', mean_y, sigma_y)
    print('R:', mean_r, sigma_r)

    if not args.vis:
      plt.ion()

    xaxis = np.arange(mean_x-5., mean_x+5., .05)
    xhist = plt.hist(dxs, xaxis)
    pars, cov = curve_fit(gauss, xhist[1][:-1] + .05, xhist[0])
    xfit = gauss(xaxis, *pars)
    plt.plot(xaxis, xfit, color='orange') 
    print(pars[1:])
    #plt.show()

    yaxis = np.arange(mean_y-5., mean_y+5., .05)
    yhist = plt.hist(dys, yaxis)
    pars, cov = curve_fit(gauss, yhist[1][:-1] + .05, yhist[0])
    yfit = gauss(yaxis, *pars)
    plt.plot(yaxis, yfit, color='orange') 
    print(pars[1:])
    #plt.show()


    raxis = np.arange(mean_r-5., mean_r+5., .05)
    rhist = plt.hist(drs, raxis)
    pars, cov = curve_fit(gauss, rhist[1][:-1] + .05, rhist[0])
    rfit = gauss(raxis, *pars)
    plt.plot(raxis, rfit, color='orange') 
    print(pars[1:])
    #plt.show()

def plot_hist(data, bins, scale=1.):
  vals, bin_edges = np.histogram(data, bins=data) 
  bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
  return vals, bin_centers


def draw(t, br_str, name, binning, cut='', scale=1.):
  draw_str = f'{br_str}>>{name}({",".join([str(s) for s in binning])})'
  t.Draw(draw_str, cut + ' && {good_event}')
  h = RT.gDirectory.Get(name)
  #h.AddBinContent(1, h.GetBinContent(0))
  #h.SetBinContent(0, 0)
  #h.AddBinContent(h.GetNbinsX(), h.GetBinContent(h.GetNbinsX()+1))
  #h.SetBinContent(h.GetNbinsX()+1, 0)
  h.Scale(scale)
  return h

def make_stack(hists, name, colors=colors):
  s = RT.THStack(name, '')
  for i, h in enumerate(hists):
    h.SetFillColor(colors[i])
    h.SetLineColor(colors[i])
    s.Add(h)
  return s

def format_data(h):
  h.SetMarkerStyle(20)
  #h.SetLineColor(RT.kBlack)

def draw_canvas(smc, hdata, name, xtitle, ytitle):
  c = RT.TCanvas(name, '')
  c.SetTicks()
  smc.Draw()
  themax = 1.1*max([smc.GetHistogram().GetMaximum(), hdata.GetMaximum()])
  hdata.SetMaximum(themax)
  hdata.Draw('pe1')
  smc.Draw('hist same')
  hdata.SetTitle('')
  hdata.GetXaxis().SetTitle(xtitle)
  hdata.GetYaxis().SetTitle(ytitle)
  hdata.SetTitleSize(.06, 'XY')
  hdata.SetTitleOffset(.8, 'XY')
  RT.gPad.RedrawAxis()

  hdata.Draw('pe1 same')
  return c

def do_compare(tMC, tData, draw_str, draw_str_data, name_base, xtitle, ytitle, binning, scale, scale_data=1.):
  hs = []
  for i in range(1, 7):
    h = draw(tMC, draw_str, f'hMC_{name_base}_{i}', binning, cut=f'cat=={i}', scale=scale)
    hs.append(h)
    h.Write()
  s = make_stack(hs, f'{name_base}_stack')
  hd = draw(tData, draw_str_data, f'hData_{name_base}', binning, scale=scale_data)
  format_data(hd)
  hd.Write()

  c = draw_canvas(s, hd, f'c{name_base}', xtitle, ytitle, )
  c.Write()


def compare_data_mc(args):
  fMC = RT.TFile.Open(args.m)
  fData = RT.TFile.Open(args.d)

  tMC = fMC.Get('tree')
  tData = fData.Get('tree')
  fOut = RT.TFile(args.o, 'recreate')

  scale = 1.e-3*tData.GetEntries()/tMC.GetEntries()
  scale_data = 1.e-3


  mcmeanx=-2.18868327
  mcsigmax= 1.15702154
  mcmeany= 0.97183832
  mcsigmay= 1.17307941
  mcmeanr= 2.71273912
  mcsigmar= 1.08169567
  datameanx= 3.0995176
  datasigmax= 1.36652396
  datameany= 1.37823363
  datasigmay= 1.60954298
  datameanr= 3.72614747
  datasigmar= 1.38783237

  draw_str = 'x'
  do_compare(tMC, tData, draw_str, draw_str, 'x',
             'Reconstructed #DeltaX at FV Face [cm]',
             'Number of Entries [x10^3]', (30, -60, 60), scale, scale_data)
  draw_str = 'x - beam_x'
  do_compare(tMC, tData, draw_str, draw_str, 'deltax',
             '#DeltaX at FV Face [cm]',
             'Number of Entries [x10^3]', (20, -20, 20), scale, scale_data)
  draw_str = f'(x - beam_x - {mcmeanx})/{mcsigmax}'
  draw_str_data = f'(x - beam_x - {datameanx})/{datasigmax}'
  do_compare(tMC, tData, draw_str, draw_str_data, 'deltax_norm',
             'Normalized #DeltaX at FV Face [cm]',
             'Number of Entries [x10^3]', (10, -5, 5), scale, scale_data)

  #cx.Write()

  #hd.Write()
  #s.Write()

  fOut.Close()
  fMC.Close()
  fData.Close()

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', required=True, type=str)
  parser.add_argument('-m', default=None, type=str)
  parser.add_argument('-d', default=None, type=str)
  parser.add_argument('-o', default='endz_effs.root', type=str)
  parser.add_argument('--vis', action='store_true')
  parser.add_argument('--routine', type=str, default=None,
                      choices=['vals', 'stats', 'compare', None])
  parser.add_argument('--n_vals', type=int, default=-1)
  args = parser.parse_args()

  if args.routine == 'vals':
    fIn = RT.TFile.Open(args.i)
    t = fIn.Get('pduneana/beamana')
    print('Here')
    get_vals(t, args.o, maxnum=args.n_vals)
    fIn.Close()
    exit()
  elif args.routine == 'stats':
    get_stats(args)
    exit()
  elif args.routine == 'compare':
    compare_data_mc(args)
    exit()
  else:
    pass

  fIn = RT.TFile.Open(args.i)
  t = fIn.Get('pduneana/beamana')

  true_topo = 'new_interaction_topology'
  reco_true_ID = 'reco_beam_true_byHits_ID'
  reco_true_origin = 'reco_beam_true_byHits_origin'
  true_ID = 'true_beam_ID'
  sel_ID = 'selection_ID'

  colors = {
    'abs':602,
    'cex':433,
    'otherinel':837,
    'cosmic':419,
    'int_to_secondary':403,
    'upstream':602,
    'muon':433,
    'pastfv':837,
    'stopping':419,
    'no_track':234,
  }
  title_names = {
    'abs':'Absorption',
  }
  good_track_cut = f' && {sel_ID} != 6 && {sel_ID} != 5'
  cuts = {
    'abs':f'{true_topo} == 1 && {reco_true_ID} == {true_ID}' + good_track_cut,
    'cex':f'{true_topo} == 2 && {reco_true_ID} == {true_ID}' + good_track_cut,
    'otherinel':f'{true_topo} == 3 && {reco_true_ID} == {true_ID}' + good_track_cut,
    'cosmic':f'{reco_true_origin} == 2 && {true_topo} < 4' + good_track_cut,
    'int_to_secondary':f'{true_topo} < 4 && {reco_true_origin} == 4 && {reco_true_ID} != {true_ID}' + good_track_cut,
    #'int_to_other':f'{true_topo} < 4 && {reco_true_origin} == 2 && {reco_true_ID} != {true_ID}' + good_track_cut,
    #'int_to_other1':f'{true_topo} < 4 && {reco_true_origin} == 4 && {reco_true_ID} != {true_ID}' + good_track_cut,

    'upstream':f'{true_topo} == 4' + good_track_cut,
    'muon':f'{true_topo} == 5' + good_track_cut,
    'pastfv':f'{true_topo} == 6' + good_track_cut,
    'stopping':f'{true_topo} == 7' + good_track_cut,
  }
  #for k,cut in cuts.items(): cut += f' && {sel_ID} != 6 && {sel_ID} != 5'
  cuts['no_track'] = f'{sel_ID} == 6'

  hs = {k:RT.TH1D(f'h_{k}', 'asdf', 100, -100, 400) for k in cuts.keys()}
  hs_reco = {k:RT.TH1D(f'h_{k}_reco', 'asdf', 100, 0, 400) for k in cuts.keys()}
  hs_calo = {k:RT.TH1D(f'h_{k}_calo', 'asdf', 100, -10, 400) for k in cuts.keys()}
  for n,cut in cuts.items():
    print('Drawing', n, cut)
    t.Draw(f'true_beam_endZ>>h_{n}', cut)
    t.Draw(f'reco_beam_endZ>>h_{n}_reco', cut)
    t.Draw(f'reco_beam_calo_endZ>>h_{n}_calo', cut)

  fout = RT.TFile(args.o, 'recreate')
  total = 0
  stack = RT.THStack()
  stack_reco = RT.THStack()
  stack_calo = RT.THStack()
  leg = RT.TLegend()
  for i, (k,h) in enumerate(hs.items()):
    print(h.Integral(0, h.GetNbinsX()+1))
    total += h.Integral(0, h.GetNbinsX()+1)
    h.SetLineColor(colors[k])
    h.SetFillColor(colors[k])
    if i >= 6: h.SetFillStyle(3144)
    h.Write()
    leg.AddEntry(h, k, 'lf')
    stack.Add(h)

  for i, (k,h) in enumerate(hs_reco.items()):
    h.SetLineColor(colors[k])
    h.SetFillColor(colors[k])
    if i >= 6: h.SetFillStyle(3144)
    h.Write()
    stack_reco.Add(h)

  for i, (k,h) in enumerate(hs_calo.items()):
    h.SetLineColor(colors[k])
    h.SetFillColor(colors[k])
    if i >= 6: h.SetFillStyle(3144)
    h.Write()
    stack_calo.Add(h)

  print(total, t.GetEntries(f'{sel_ID} != 5 && {sel_ID} != 6'), t.GetEntries(f'{sel_ID} != 5'))
  stack.Write("hstack")
  stack_reco.Write("hstack_reco")
  stack_calo.Write("hstack_calo")
  leg.Write('leg')
  fout.Close()


  effs, purs = [], []
 # np.zeros((t.GetEntries(), len(np.arange(0., 75., 5.)) + 1))
  for i in np.arange(0., 75., 5.):

    '''base_signal = (e.true_beam_endZ > 0. and e.true_beam_endProcess == "pi+Inelastic" and e.new_interaction_topology != 6)
    base_selected = (e.primary_isBeamType and e.primary_ends_inAPA3 and not e.vertex_cut)

      in_fv = e.reco_beam_calo_endZ > i 
      index = np.where(e.reco_beam_calo_Z >= i)[0][0]'''
    base_signal = f'true_beam_endZ > {i} && true_beam_endProcess == \"pi+Inelastic\" && {true_topo} != 6'
    base_selection = f'reco_beam_calo_endZ > {i} && reco_beam_calo_endZ < 230. && {sel_ID} != 5 && {sel_ID} != 6'
    true_signal = t.GetEntries(base_signal)
    true_selected = t.GetEntries(f'{base_signal} && {base_selection} && {reco_true_ID} == {true_ID}')
    false_selected = t.GetEntries(f'!({base_signal} && {reco_true_ID} == {true_ID}) && {base_selection}')
    print(false_selected, true_selected)

    effs.append(true_selected/true_signal)
    purs.append(true_selected/(true_selected + false_selected))


  effs = np.array(effs)
  purs = np.array(purs)
  print(effs)
  print(purs)

  xs = np.arange(0., 75., 5.)
  if args.vis:
    fig, ax = plt.subplots()
    #ax.plot(t, s)

    ax.plot(xs, effs, label='Efficiency')
    ax.plot(xs, purs, label='Purity')
    ax.plot(xs, effs*purs, label='p*e')
    ax.set(xlabel='End Z cut (cm)',
           title='1 GeV Pions')
    plt.legend()
    #plt.ylim(0., 1.)
    plt.show()
  fIn.Close()
