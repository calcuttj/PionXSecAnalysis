import ROOT as RT
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser as ap

def draw_eff_pur(eff, purity, labels):
  fig, ax = plt.subplots(layout='constrained') #
  plt.ylim(0., 1.2)
  pur_line = ax.plot(purity, label='Purity')
  eff_line = ax.plot(eff, label='Efficiency')
  pur_eff = eff*purity
  pe_line = ax.plot(pur_eff, label='Pur. x Eff.')
  ax.set_xticks(np.arange(len(labels)), labels=labels)
  ax.legend()
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
  #plt.show()



def run(args):
  RT.gROOT.SetBatch()
  filename = args.i
  f = RT.TFile.Open(filename)
  t = f.Get('pduneana/beamana')

  cuts = [
    'selection_ID != 6',
  ]
  add = (' && true_beam_endZ > 30.' if args.fv else '')
  add += (' && true_beam_endP > .6241554 && true_beam_startP < .929145' if args.ke else '')
  cuts[0] += add
  cuts.append(cuts[-1] + ' && reco_beam_calo_endZ > 30.')
  cuts.append(cuts[-1] + ' && selection_ID != 5')
  cuts.append(cuts[-1] + ' && selection_ID != 4')
  if not args.no_michel:
    cuts.append(cuts[-1] + ' && selection_ID != 7')

  print(cuts)

  abs_cut = cuts[-1] + ' && selection_ID == 1'
  cex_cut = cuts[-1] + ' && selection_ID == 2'
  other_cut = cuts[-1] + ' && selection_ID == 3'
  

  total_abs = t.GetEntries('new_interaction_topology == 1' + add)
  total_cex = t.GetEntries('new_interaction_topology == 2' + add)
  total_other = t.GetEntries('new_interaction_topology == 3' + add)

  sels = [t.GetEntries()] + [t.GetEntries(c) for c in cuts]

  sel_abs = sels + [t.GetEntries(abs_cut)]
  sel_cex = sels + [t.GetEntries(cex_cut)]
  sel_other = sels + [t.GetEntries(other_cut)]

  true_abs_sel_abs = np.array(
    [total_abs] +
    [t.GetEntries(c + ' && new_interaction_topology == 1') for c in cuts] +
    [t.GetEntries(abs_cut + ' && new_interaction_topology == 1')]
  )

  true_cex_sel_cex = np.array(
    [total_cex] +
    [t.GetEntries(c + ' && new_interaction_topology == 2') for c in cuts] +
    [t.GetEntries(cex_cut + ' && new_interaction_topology == 2')]
  )

  true_other_sel_other = np.array(
    [total_other] +
    [t.GetEntries(c + ' && new_interaction_topology == 3') for c in cuts] +
    [t.GetEntries(other_cut + ' && new_interaction_topology == 3')]
  )

  print(sels)

  abs_purity = true_abs_sel_abs/sel_abs 
  abs_eff = true_abs_sel_abs/total_abs
  print(abs_purity)
  print(true_abs_sel_abs/total_abs)

  cex_purity = true_cex_sel_cex/sel_cex 
  cex_eff = true_cex_sel_cex/total_cex
  print(cex_purity)
  print(cex_eff)

  other_purity = true_other_sel_other/sel_other 
  other_eff = true_other_sel_other/total_other
  print(other_purity)
  print(true_other_sel_other/total_other)

  labels = [
    'All', 'Beam Track', 'FV Low End', 'Beam Consistency',
    'Within FV', 'No Michel Vertex']
  if args.no_michel: labels.pop(-1)

  #fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
  draw_eff_pur(abs_eff, abs_purity, labels=(labels+['Absorption']))
  plt.title('Absorption')
  plt.savefig('abs_eff_pur.pdf')
  plt.savefig('abs_eff_pur.png')
  plt.close()
  draw_eff_pur(cex_eff, cex_purity, labels=(labels+['Ch. Exch.']))
  plt.title('Charge Exchange')
  plt.savefig('cex_eff_pur.pdf')
  plt.savefig('cex_eff_pur.png')
  plt.close()
  draw_eff_pur(other_eff, other_purity, labels=(labels+['Other']))
  plt.title('Other')
  plt.savefig('other_eff_pur.pdf')
  plt.savefig('other_eff_pur.png')
  plt.close()

  '''delta_E = 'reco_beam_fixedinteractingEnergy - (sqrt(true_beam_endP*true_beam_endP*1.e6 + 139.57*139.57) - 139.57)'
  t.Draw(delta_E + > 'htotal_abs', ('new_interaction_topology == 1' + add))

  true_abs_sel_abs = np.array(
    [total_abs] +
    [t.GetEntries(c + ' && new_interaction_topology == 1') for c in cuts] +
    [t.GetEntries(abs_cut + ' && new_interaction_topology == 1')]
  )'''


if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', type=str, required=True)
  parser.add_argument('--fv', action='store_true')
  parser.add_argument('--no_michel', action='store_true')
  parser.add_argument('--ke', action='store_true')
  args = parser.parse_args()

  run(args)
