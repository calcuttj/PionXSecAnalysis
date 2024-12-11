import ROOT as RT
from argparse import ArgumentParser as ap
RT.gROOT.SetBatch()

def draw(t, var='true_beam_startP', binning='(100,0,2)', pdg=211):
  colors = [RT.kRed, RT.kBlue, RT.kGreen, RT.kMagenta]
  cuts = [6, 5, 4, -1]
  hs = []
  for cut, color in zip(cuts, colors):
    name = f'h{(cut if cut != -1 else "Other")}_{abs(pdg)}' 
    cut_str = f'selection_ID == {cut}' if cut != -1 else '(selection_ID < 4  || selection_ID == 7)'
    t.Draw(f'{var}>>{name}{binning}',
           (f'true_beam_is_scraper && true_beam_PDG == {pdg} && ' + cut_str))
  
    h = RT.gDirectory.Get(name)
    h.SetFillColor(color)
    h.SetLineColor(color)
    hs.append(h)
  return hs

def make_stack(hs, name):
  s = RT.THStack(name, '')
  for h in hs:
    print(h)
    s.Add(h)
  return s

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', type=str)
  parser.add_argument('-o', type=str)

  args = parser.parse_args()

  f = RT.TFile.Open(args.i)
  t = f.Get('pduneana/beamana')
  fout = RT.TFile(args.o, 'recreate')

  hpis = draw(t)
  pistack = make_stack(hpis, 'spi')
  pistack.Draw()
  pistack.GetHistogram().SetTitle('Pion Scrapers;True Initial Momentum (GeV/c);Entries')
  pistack.Write()
  for h in hpis: h.Write()
  leg = RT.TLegend()
  leg_labels = ['No Track', 'Beam Cut', 'Past FV', 'Other']
  for h, l in zip(hpis, leg_labels):
    leg.AddEntry(h, l) 
  leg.Write('leg')

  hmus = draw(t, pdg=-13)
  for h in hmus: h.Write()
  mustack = make_stack(hmus, 'smu')
  mustack.Draw()
  mustack.GetHistogram().SetTitle('Muon Scrapers;True Initial Momentum (GeV/c);Entries')
  mustack.Write()

  hmus = draw(t, var='true_beam_endZ', binning='(200, 0, 400)', pdg=-13)
  for h in hmus: h.Write()
  mustack = make_stack(hmus, 'smu_endz')
  mustack.Draw()
  mustack.GetHistogram().SetTitle('Muon Scrapers;True End Z (cm);Entries')
  mustack.Write()
  f.Close()
