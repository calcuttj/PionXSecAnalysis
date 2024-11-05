import ROOT as RT
RT.gStyle.SetOptStat(0)
RT.gROOT.SetBatch()
import numpy as np
from argparse import ArgumentParser as ap

def get_leading(momenta, indices):
  return max(momenta[indices]) if len(indices[0]) > 0 else -999.

def get_leading_costheta(pz, momentum, indices):
  leading_costheta = get_leading(pz, indices)
  return (-999. if momentum < 0.
          else leading_costheta/momentum)

def check_above_thresh(momenta, indices):
  if len(indices[0]) == 0: return False

  return (max(momenta[indices]) > .150)

def process(args):
  f = RT.TFile.Open(args.i)
  fOut = RT.TFile(args.o, 'recreate')
  t = f.Get('ginuke')
  tOut = RT.TTree('tree', '')
  br_leading_pi0_costheta = np.array([-999.])
  br_leading_p_costheta = np.array([-999.])
  br_leading_piplus_costheta = np.array([-999.])
  br_leading_pi0_momentum = np.array([-999.])
  br_leading_p_momentum = np.array([-999.])
  br_leading_piplus_momentum = np.array([-999.])
  br_ke = np.array([-999.])
  br_has_pi_thresh = np.array([0])
  br_topo = np.array([0])
  br_n_neutron = np.array([0])
  br_n_proton = np.array([0])

  tOut.Branch('leading_pi0_costheta', br_leading_pi0_costheta,
              'leading_pi0_costheta/D')
  tOut.Branch('leading_pi0_momentum', br_leading_pi0_momentum,
              'leading_pi0_momentum/D')
  tOut.Branch('leading_piplus_costheta', br_leading_piplus_costheta,
              'leading_piplus_costheta/D')
  tOut.Branch('leading_piplus_momentum', br_leading_piplus_momentum,
              'leading_piplus_momentum/D')
  tOut.Branch('leading_p_costheta', br_leading_p_costheta,
              'leading_p_costheta/D')
  tOut.Branch('leading_p_momentum', br_leading_p_momentum,
              'leading_p_momentum/D')
  tOut.Branch('has_pi_thresh', br_has_pi_thresh, 'has_pi_thresh/I')
  tOut.Branch('ke', br_ke, 'ke/D')
  tOut.Branch('topology', br_topo, 'topology/I')
  tOut.Branch('n_neutron', br_n_neutron, 'n_neutron/I')
  tOut.Branch('n_proton', br_n_proton, 'n_proton/I')

  for e in t:
    if e.probe_fsi == 1: continue
    
    pdgs = np.array(e.pdgh)
    pi0_indices = np.where(pdgs == 111)
    piplus_indices = np.where(pdgs == 211)
    pic_indices = np.where(abs(pdgs) == 211)
    p_indices = np.where(pdgs == 2212)
    n_indices = np.where(pdgs == 2112)

    momenta = np.array(e.ph)
    pz = np.array(e.pzh)
    leading_pi0_momentum = get_leading(momenta, pi0_indices) 
    leading_piplus_momentum = get_leading(momenta, piplus_indices) 
    leading_p_momentum = get_leading(momenta, p_indices) 

    leading_pi0_costheta = get_leading_costheta(
        pz, leading_pi0_momentum, pi0_indices)
    leading_piplus_costheta = get_leading_costheta(
        pz, leading_piplus_momentum, piplus_indices)
    leading_p_costheta = get_leading_costheta(
        pz, leading_p_momentum, p_indices)

    br_leading_pi0_costheta[0] = leading_pi0_costheta
    br_leading_pi0_momentum[0] = leading_pi0_momentum

    br_leading_piplus_costheta[0] = leading_piplus_costheta
    br_leading_piplus_momentum[0] = leading_piplus_momentum

    br_leading_p_costheta[0] = leading_p_costheta
    br_leading_p_momentum[0] = leading_p_momentum
    br_has_pi_thresh[0] = check_above_thresh(momenta, pic_indices)
    print(pdgs, br_has_pi_thresh[0])
    print(momenta, leading_piplus_momentum, leading_piplus_costheta)

    br_ke[0] = e.ke

    br_topo[0] = (3 if br_has_pi_thresh[0] else 1)
    if len(pi0_indices[0]) > 0 and br_topo[0] == 1: br_topo[0] += 1

    br_n_neutron[0] = len(n_indices[0])
    br_n_proton[0] = len(p_indices[0])

    tOut.Fill()


  tOut.Write()
  fOut.Close()
  f.Close()

def draw_costheta(t, part):
  h_leading_costheta = RT.TH2D(
      f'h_leading_{part}_costheta', '', 25, -1, 1.0, 25, 0., 1.)
  t.Draw(f'ke:leading_{part}_costheta>>h_leading_{part}_costheta',
         f'leading_{part}_momentum > 0.')
  return h_leading_costheta

def scale(h, h_scale):
  for i in range(1, h.GetNbinsX()+1):
    for j in range(1, h.GetNbinsY()+1):
      content = h.GetBinContent(i, j)
      h.SetBinContent(
        i, j, content*h_scale.GetBinContent(j)
      )


def draw(args):
  f = RT.TFile.Open(args.i)
  t = f.Get('tree')
  fOut = RT.TFile(args.o, 'recreate')

  h_leading_piplus_costheta = draw_costheta(t, 'piplus')
  h_leading_p_costheta = draw_costheta(t, 'p')
  h_leading_pi0_costheta = draw_costheta(t, 'pi0')

  if args.ke_events is not None:
    h_ke = RT.TH1D('h_ke', '', 25, 0., 1.)
    t.Draw('ke>>h_ke')

    f_ke_events = RT.TFile.Open(args.ke_events)
    h_ke_events = f_ke_events.Get('ke_events')

    h_ke_events.Divide(h_ke)
    fOut.cd()    
    h_ke_events.Write('h_scale')
    h_ke_events.SetDirectory(0)
    f_ke_events.Close()

    scale(h_leading_piplus_costheta, h_ke_events)
    scale(h_leading_pi0_costheta, h_ke_events)
    scale(h_leading_p_costheta, h_ke_events)

  h_leading_piplus_costheta.ProjectionX().Write('h_leading_piplus_costheta')
  h_leading_p_costheta.ProjectionX().Write('h_leading_p_costheta')
  h_leading_pi0_costheta.ProjectionX().Write('h_leading_pi0_costheta')
  fOut.Close()

  f.Close()

def get_complex(proj, t, args, part, topo, ike):


  name = (
    f'h_leading_{part}_{proj}_ke{ike-1}_topo{topo}' if topo > 0
    else f'h_leading_{part}_{proj}_ke{ike-1}_topo0'
  )
 
  masses = {
    'piplus':0.13957,
    'p':0.93827,
    'pi0':0.135,
  }
  m = masses[part]
  if part == 'p':
    end = np.sqrt((args.kes[ike] + masses['piplus'] + m)**2 - m**2)
  else:
    end = np.sqrt((args.kes[ike]+m)**2 - m**2)
  binning = ((25, -1., 1.) if proj == 'costheta' else (25, 0., end))
  h = RT.TH1D(name, '', *binning)
  t.Draw(f'leading_{part}_{proj}>>{name}',
         (f'leading_{part}_momentum > 0.'
          f' && ke > {args.kes[ike-1]} && ke < {args.kes[ike]}'
          + (f' && topology == {topo}' if topo > 0
             else ' && topology < 4')))
  if h.Integral() > 0.:# and args.scale:
    h.Scale(1./h.Integral())

  return h

def get_complex_pdsp(proj, t, args, part, topo, ike):
  name = (
    f'h_leading_{part}_{proj}_ke{ike-1}_topo{topo}' if topo > 0
    else f'h_leading_{part}_{proj}_ke{ike-1}_topo0'
  )
 

  #name = f'h_leading_{part}_{proj}_ke{ike-1}_topo{topo}'
  masses = {
    'piplus':0.13957,
    'p':0.93827,
    'pi0':0.135,
  }
  m = masses[part]
  if part == 'p':
    end = np.sqrt((args.kes[ike] + masses['piplus'] + m)**2 - m**2)
  else:
    end = np.sqrt((args.kes[ike]+m)**2 - m**2)
  binning = ((25, -1., 1.) if proj == 'costheta' else (25, 0., end))
  h = RT.TH1D(name, '', *binning)
  ke = 'sqrt(true_beam_endP*true_beam_endP + .13957*.13957) - .13957'
  t.Draw(f'leading_{part}_{proj}>>{name}',
         (f'leading_{part}_momentum > 0.'
          f' && {ke} > {args.kes[ike-1]} && {ke} < {args.kes[ike]}'
          + (f' && new_interaction_topology == {topo}' if topo > 0
             else ' && new_interaction_topology < 4')))
  if h.Integral() > 0.:#and args.scale:
    h.Scale(1./h.Integral())

  return h



def process_pdsp(args):
  #Open PDSP file
  f = RT.TFile.Open(args.i)
  t = f.Get('pduneana/beamana')
  fOut = RT.TFile(args.o, 'recreate')

  parts = ['piplus', 'p', 'pi0']
  for proj in ('costheta', 'momentum'):
    for part in parts:
      for topo in (range(1) if args.total else range(1, 4)):
        for ike in range(1, len(args.kes)):
          h = get_complex_pdsp(proj, t, args, part, topo, ike)
          h.Write()
  fOut.Close()
  f.Close()

def get_g4(proj, t, args, part, topo, ike):
  name = (
    f'h_leading_{part}_{proj}_ke{ike-1}_topo{topo}' if topo > 0
    else f'h_leading_{part}_{proj}_ke{ike-1}_topo0'
  )
 
  masses = {
    'piplus':139.57,
    'p':938.27,
    'pi0':135.0,
  }
  m = masses[part]
  if part == 'p':
    end = np.sqrt((args.kes[ike] + masses['piplus'] + m)**2 - m**2)
  else:
    end = np.sqrt((args.kes[ike]+m)**2 - m**2)
  binning = ((25, -1., 1.) if proj == 'costheta' else (25, 0., end))
  h = RT.TH1D(name, '', *binning)
  ke = 'sqrt(momentum*momentum + .13957*.13957) - .13957'
  cuts = {
    1:('c_leading_piplus_momentum < 150. && '
       'c_leading_piminus_momentum < 150. && nPi0 == 0'),
    2:('c_leading_piplus_momentum < 150. && '
       'c_leading_piminus_momentum < 150. && nPi0 > 0'),
    3:('c_leading_piplus_momentum >= 150. || '
       'c_leading_piminus_momentum >= 150.'),
  }
  if part == 'p': part = 'proton'
  t.Draw(f'c_leading_{part}_{proj}>>{name}',
         (f'c_leading_{part}_momentum > 0.'
          f' && {ke} > {args.kes[ike-1]} && {ke} < {args.kes[ike]}'
          + (f' && ({cuts[topo]})' if topo > 0
             else '')))
  if h.Integral() > 0.:#and args.scale:
    h.Scale(1./h.Integral())

  return h



def process_g4(args):
  f = RT.TFile.Open(args.i)
  t = f.Get('tree')
  fOut = RT.TFile(args.o, 'recreate')

  parts = ['piplus', 'p', 'pi0']
  for proj in ('costheta', 'momentum'):
    for part in parts:
      for topo in (range(1) if args.total else range(1, 4)):
        for ike in range(1, len(args.kes)):
          h = get_g4(proj, t, args, part, topo, ike)
          h.Write()
  fOut.Close()
  f.Close()


def complex_draw(args):
  #Open GENIE file
  f = RT.TFile.Open(args.i)
  t = f.Get('tree')
  fOut = RT.TFile(args.o, 'recreate')

  parts = ['piplus', 'p', 'pi0']
  for proj in ('costheta', 'momentum'):
    for part in parts:
      if args.total:
        for ike in range(1, len(args.kes)):
          h = get_complex(proj, t, args, part, 0, ike)
          h.Write()
      else:
        for topo in range(1, 4):
          for ike in range(1, len(args.kes)):
            h = get_complex(proj, t, args, part, topo, ike)
            h.Write()
  fOut.Close()
  f.Close()

def compare(args):
  RT.gStyle.SetOptStat(0)
  fPDSP = RT.TFile.Open(args.pdsp)
  fGenie = RT.TFile.Open(args.genie)

  fOut = RT.TFile(args.o, 'recreate')
  parts = ['piplus', 'p', 'pi0']
  for proj in ('costheta', 'momentum'):
    for part in parts:
      for topo in (range(1) if args.total else range(1, 4)):
        for ike in range(0, args.nke):
          h_pdsp = fPDSP.Get(f'h_leading_{part}_{proj}_ke{ike}_topo{topo}')
          h_genie = fGenie.Get(f'h_leading_{part}_{proj}_ke{ike}_topo{topo}')

          c = RT.TCanvas(f'c_{part}_{proj}_ke{ike}_topo{topo}')
          c.SetTicks()

          h_genie.SetLineColor(RT.kRed)
          themax = max([h_pdsp.GetMaximum(), h_genie.GetMaximum()])
          if h_pdsp.GetMaximum() < h_genie.GetMaximum():
            h_pdsp.SetMaximum(1.1*h_genie.GetMaximum())
          h_pdsp.SetMinimum(0.)
          h_pdsp.Draw('hist')
          h_genie.Draw('hist same')

          c.Write()

  fOut.Close()
  fPDSP.Close()
  fGenie.Close()


def expand_genie(h_pdsp, h_genie, expand=None):
  if expand is None: return

  h_sub = h_pdsp.Clone('h_sub')
  h_sub.Scale(-1.)

  h_delta = h_genie.Clone('h_delta')
  h_delta.Add(h_sub)

  h_delta.Scale(expand - 1.)

  h_genie.Add(h_delta)

def get_delta(h_pdsp, h_genie):
  h_sub = h_pdsp.Clone('h_sub')
  h_sub.Scale(-1.)
  h_delta = h_genie.Clone('h_delta')
  h_delta.Add(h_sub)

  h_prime = h_pdsp.Clone('h_prime')
  h_delta.Scale(-1.)
  h_prime.Add(h_delta)
  for i in range(1, h_prime.GetNbinsX()+1):
    if h_prime.GetBinContent(i) < 0: h_prime.SetBinContent(i, 0)
    if h_pdsp.GetBinContent(i) < 1.e-5: h_prime.SetBinContent(i, 0)

  if h_prime.Integral() > 0.:
    h_prime.Scale(1./h_prime.Integral())
  return h_prime

def get_ratio(h_target, h_base):
  h_ratio = h_target.Clone()
  total = 0.
  for i in range(1, h_target.GetNbinsX()+1):
    target = h_target.GetBinContent(i)
    base = h_base.GetBinContent(i)
    if base < 1.e-5:
      ratio = 0. 
    elif target/base > 5.:
      ratio = 5.
    else:
      ratio = target/base 
    h_ratio.SetBinContent(i, ratio)
    total += ratio*h_base.GetBinContent(i)

  if (total > 0.): h_ratio.Scale(1./total)
  new_total = 0.
  for i in range(1, h_ratio.GetNbinsX()+1):
    new_total += h_ratio.GetBinContent(i)*h_base.GetBinContent(i)
  print(f'{new_total:.2f}', h_ratio.GetName())
  return h_ratio


def make_vars(args):
  RT.gStyle.SetOptStat(0)
  fPDSP = RT.TFile.Open(args.pdsp)
  fGenie = RT.TFile.Open(args.genie)

  titles = {
    'piplus':'Leading-p #pi^{+} ',
    'pi0':'Leading-p #pi^{0} ',
    'p':'Leading-p p ',
  }

  fOut = RT.TFile(args.o, 'recreate')
  parts = ['piplus', 'p', 'pi0']
  for proj in ('costheta', 'momentum'):
    for part in parts:
      for topo in (range(1) if args.total else range(1, 4)):
        for ike in range(0, args.nke):
          title = titles[part]
          title += (f'| Incident KE ({args.kes[ike]}, {args.kes[ike+1]}) MeV;')
          var = "cos#theta" if proj == "costheta" else "p MeV/c"
          title += f'{var};Fraction of Events'

          h_pdsp = fPDSP.Get(f'h_leading_{part}_{proj}_ke{ike}_topo{topo}')
          h_pdsp.SetTitle(title)
          h_genie = fGenie.Get(f'h_leading_{part}_{proj}_ke{ike}_topo{topo}')
          #expand_genie(h_pdsp, h_genie, args.expand)
          h_prime = get_delta(h_pdsp, h_genie)

          #h_plus = h_genie.Clone(
          #    'h_leading_{part}_{proj}_ke{ike}_topo{topo}_plus')
          #h_plus.Divide(h_pdsp)
          #h_minus = h_prime.Clone(
          #    'h_leading_{part}_{proj}_ke{ike}_topo{topo}_minus')
          #h_minus.Divide(h_pdsp)
          h_plus = get_ratio(h_genie, h_pdsp)
          h_minus = get_ratio(h_prime, h_pdsp)
          h_plus.Write(f'h_plus_leading_{part}_{proj}_ke{ike}_{topo}')
          h_minus.Write(f'h_minus_leading_{part}_{proj}_ke{ike}_{topo}')

          c = RT.TCanvas(f'c_{part}_{proj}_ke{ike}_topo{topo}')
          c.SetTicks()

          h_genie.SetLineColor(RT.kRed)
          themax = max([
            h_pdsp.GetMaximum(),
            h_genie.GetMaximum(),
            h_prime.GetMaximum(),
          ])
          h_pdsp.SetMaximum(1.1*themax)
          h_pdsp.SetMinimum(0.)
          h_pdsp.Draw('hist')
          h_genie.Draw('hist same')
          h_prime.SetLineColor(RT.kBlack)
          h_prime.Draw('hist same')
          #h_prime.Write(f'h_leading_{part}_{proj}_ke{ike}_topo{topo}_prime')

          c.Write()

  fOut.Close()
  fPDSP.Close()
  fGenie.Close()


def get_ke(e):
  return 1.e-3*(np.sqrt(
    e.true_beam_endP*e.true_beam_endP*1.e6 + 139.57**2
  ) - 139.57)

def weight_pdsp(args):
  fPDSP = RT.TFile.Open(args.pdsp)
  t = fPDSP.Get('pduneana/beamana')

  fVar = RT.TFile.Open(args.i)
  topos = range(1) if args.total else range(1, 4)
  kes = range(args.nke)
  parts = ['piplus', 'p', 'pi0']
  plus_hists = {
    (topo, ke, p): fVar.Get(f'h_plus_leading_{p}_costheta_ke{ke}_{topo}')
    for topo in topos for ke in kes for p in parts
  }
  minus_hists = {
    (topo, ke, p): fVar.Get(f'h_minus_leading_{p}_costheta_ke{ke}_{topo}')
    for topo in topos for ke in kes for p in parts
  }
 
  fOut = RT.TFile(args.o, 'recreate')
  hs_out_plus = {
    p:RT.TH1D(f'h_leading_{p}_costheta_plus', '', 25, -1., 1.) for p in parts
  }
  hs_out_minus = {
    p:RT.TH1D(f'h_leading_{p}_costheta_minus', '', 25, -1., 1.) for p in parts
  }
  hs_out_nom = {
    p:RT.TH1D(f'h_leading_{p}_costheta_nom', '', 25, -1., 1.) for p in parts
  }

  for e in t:
    topo = e.new_interaction_topology
    if topo > 3: continue
    if args.total: topo = 0

    for p in parts:
      if p == 'piplus':
        val = e.leading_piplus_costheta
      elif p == 'pi0':
        val = e.leading_pi0_costheta
      else:
        val = e.leading_p_costheta

      if val < -100.: continue

      ke = get_ke(e)
      if ke > args.kes[-1]:
        weight = 1.
      else:
        for ike in range(1, len(args.kes)):
          if ke > args.kes[ike-1] and ke < args.kes[ike]:
            break
        h = plus_hists[(topo, ike-1, p)]
        weight_plus = h.GetBinContent(h.FindBin(val))

        h = minus_hists[(topo, ike-1, p)]
        weight_minus = h.GetBinContent(h.FindBin(val))

      hs_out_plus[p].Fill(val, weight_plus)
      hs_out_minus[p].Fill(val, weight_minus)
      hs_out_nom[p].Fill(val)
  fOut.cd()
  for h in hs_out_plus.values(): h.Write()
  for h in hs_out_minus.values(): h.Write()
  for h in hs_out_nom.values(): h.Write()
  fPDSP.Close()
  fVar.Close()

def compare_multiple(args):
  RT.gStyle.SetOptStat(0)
  fPDSP = RT.TFile.Open(args.pdsp)
  fAlts = [RT.TFile.Open(f) for f in args.alts]

  fOut = RT.TFile(args.o, 'recreate')
  parts = ['piplus', 'p', 'pi0']
  titles = {
    'piplus':'Leading-p #pi^{+} ',
    'pi0':'Leading-p #pi^{0} ',
    'p':'Leading-p p ',
  }
  colors = [RT.kRed, RT.kBlack]
  leg = RT.TLegend()
  add_to_leg = True
  for proj in ('costheta', 'momentum'):
    for part in parts:
      for topo in (range(1) if args.total else range(1, 4)):
        for ike in range(0, args.nke):
          h_pdsp = fPDSP.Get(f'h_leading_{part}_{proj}_ke{ike}_topo{topo}')
          title = titles[part]
          title += (f'| Incident KE ({args.kes[ike]}, {args.kes[ike+1]}) MeV;')
          var = "cos#theta" if proj == "costheta" else "p MeV/c"
          title += f'{var};Fraction of Events'
          h_pdsp.SetTitle(title)
          name = f'h_leading_{part}_{proj}_ke{ike}_topo{topo}'
          h_alts = [f.Get(name) for f in fAlts]

          c = RT.TCanvas(f'c_{part}_{proj}_ke{ike}_topo{topo}')
          c.SetTicks()

          for i, h in enumerate(h_alts):
            h.SetLineColor(colors[i])
          themax = max([h_pdsp.GetMaximum()] + [h.GetMaximum() for h in h_alts])
          #if h_pdsp.GetMaximum() < h_genie.GetMaximum():
          #  h_pdsp.SetMaximum(1.1*h_genie.GetMaximum())
          h_pdsp.SetMaximum(1.1*themax)
          h_pdsp.SetMinimum(0.)
          h_pdsp.Draw('hist')
          for h in h_alts: h.Draw('hist same')
          #h_genie.Draw('hist same')

          if add_to_leg:
            leg.AddEntry(h_pdsp, 'G4 Bertini', 'l')
            for i, h in enumerate(h_alts):
              leg.AddEntry(h, args.alt_names[i], 'l')
            leg.Write('leg')
            add_to_leg = False

          c.Write()

  fOut.Close()
  fPDSP.Close()
  for f in fAlts: f.Close()


def average(args):
  fAlts = [RT.TFile.Open(f) for f in args.alts]

  fPDSP = RT.TFile.Open(args.pdsp)
  fOut = RT.TFile(args.o, 'recreate')
  parts = ['piplus', 'p', 'pi0']
  #colors = [RT.kRed, RT.kBlack]
  for proj in ('costheta', 'momentum'):
    for part in parts:
      for topo in (range(1) if args.total else range(1, 4)):
        for ike in range(0, args.nke):
          name = f'h_leading_{part}_{proj}_ke{ike}_topo{topo}'
          print(name)
          h_pdsp = fPDSP.Get(name)
          print(h_pdsp)
          h_new = h_pdsp.Clone()
          h_new.Reset()

          h_alts = [f.Get(name) for f in fAlts]

          for i in range(1, h_new.GetNbinsX()+1):
            content = sum([h.GetBinContent(i) for h in h_alts])/len(h_alts)
            h_new.SetBinContent(i, content)
          if h_new.Integral() > 0.:
            h_new.Scale(1./h_new.Integral())
          h_new.Write()

  fOut.Close()
  fPDSP.Close()
  for f in fAlts: f.Close()
  

def save(args):
  f = RT.TFile.Open(args.i)
  parts = [
    ('piplus', 3),
    ('pi0', 2),
    ('p', 1),
  ]
  add = 'vars' if args.routine == 'save_vars' else ''
  if args.routine == 'save_vars':
    h_plus = RT.TH1D('h_plus', '', 1, 0, 1)
    h_minus = RT.TH1D('h_minus', '', 1, 0, 1)
    h = RT.TH1D('h', '', 1, 0, 1)
    h_plus.SetLineColor(RT.kRed)
    h_minus.SetLineColor(RT.kBlack)
    leg = RT.TLegend(.15, .6, .4, .85)
    leg.AddEntry(h, 'Nominal', 'l')
    leg.AddEntry(h_plus, '+1#sigma', 'l')
    leg.AddEntry(h_minus, '-1#sigma', 'l')

  for proj in ['costheta', 'momentum']:
    for i in range(args.nke):
      for p, j in parts:
        c = f.Get(f'c_{p}_{proj}_ke{i}_topo{j}')
        c.Draw()
        if i == 0:
          leg.Draw('same')
        c.SaveAs(f'{p}_{proj}_ke{i}_topo{j}_{add}.pdf')
  f.Close()

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', type=str)
  parser.add_argument('-o', type=str, default='genie_kinematics.root')
  parser.add_argument('--pdsp', type=str, default='pdsp.root')
  parser.add_argument('--genie', type=str, default='genie.root')
  parser.add_argument('--alts', type=str, nargs='+')
  parser.add_argument('--alt_names', type=str, nargs='+')
  parser.add_argument('--routine', type=str,
                      default='process',
                      choices=['process', 'draw', 'complex', 'process_pdsp',
                               'process_g4',
                               'compare', 'make_vars', 'weight_pdsp',
                               'compare_multi', 'average',
                               'save', 'save_vars',
                      ])
  parser.add_argument('--nke', type=int, default=5)
  parser.add_argument('--ke_events', type=str, default=None)
  parser.add_argument('--kes', nargs='+', type=float)
  #parser.add_argument('--scale', action='store_true')
  parser.add_argument('--total', action='store_true')
  parser.add_argument('--expand', type=float, default=None)
  parser.add_argument('--minus', action='store_true')
  args = parser.parse_args()

  routines = {
    'process':process,
    'draw':draw,
    'complex':complex_draw,
    'process_pdsp':process_pdsp,
    'process_g4':process_g4,
    'compare':compare,
    'make_vars':make_vars,
    'weight_pdsp':weight_pdsp,
    'compare_multi':compare_multiple,
    'average':average,
    'save':save,
    'save_vars':save,
  }

  routines[args.routine](args)
