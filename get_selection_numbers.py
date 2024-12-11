import ROOT as RT
import sys
from argparse import ArgumentParser as ap

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-m', type=str, required=True)
  parser.add_argument('-d', type=str, required=True)
  parser.add_argument('--no_michel', action='store_true')
  args = parser.parse_args()

fMC = RT.TFile.Open(args.m)
fData = RT.TFile.Open(args.d)

tMC = fMC.Get('pduneana/beamana')
tData = fData.Get('pduneana/beamana')

beam_p_cut = 'beam_inst_P > 0.75 && beam_inst_P < 1.25'
total_mc = tMC.GetEntries()
total_data = tData.GetEntries(beam_p_cut)

print('\tMC, Data')
print('Totals', total_mc, total_data)
selected_mc = []
lines = {
  'total':f'Total & {total_mc} & {total_data} & 100 & 100 \\\\'
}
order = ['no_beam', 'beam_qual', 'ediv', 'michel', 'abs', 'cex', 'other', 'total']
if args.no_michel:
  order.pop(order.index('michel'))

labels = {
  'no_beam':'No Reco. Beam Track',
  'beam_qual':'Inconsistent with Beam',
  'ediv':'Past Electron Diverters',
  'michel':'Michel Vertex',
  'abs':'Absorption',
  'cex':'Charge Exchange',
  'other':'Other',
  'bg':'Background',
}

sets = {
 1:'abs',
 2:'cex',
 3:'other',
 4:'ediv',
 5:'beam_qual',
 6:'no_beam',
 7:'michel',
}

if args.no_michel:
  sets.pop(7)

for i in range(1, (7 if args.no_michel else 8)):
  nmc = tMC.GetEntries(f'selection_ID == {i}')
  ndata = tData.GetEntries(f'selection_ID == {i} && ' + beam_p_cut)
  #print(i, nmc, ndata, f'{100.*nmc/total_mc:.2f}', f'{100.*ndata/total_data:.2f}')
  lines[sets[i]] = f'{labels[sets[i]]} & {nmc} & {ndata} & {100.*nmc/total_mc:.2f} & {100.*ndata/total_data:.2f} \\\\'

for o in order:
  print(lines[o])

print()
print('Efficiencies')
eff_lines = {
}
order = ['abs', 'cex', 'other', 'bg']
sets = {
  1:'abs',
  2:'cex',
  3:'other',
  4:'bg',
}

labels['bg'] = 'Rejected'
for i in range(1, 5):
  truth_cut = (f'new_interaction_topology == {i}' if (i < 4)
               else 'new_interaction_topology > 3')
  n_true = tMC.GetEntries(truth_cut)

  eff_lines[sets[i]] = f'{labels[sets[i]]} '
  for j in range(1, 5):
    sel_cut = (f'selection_ID == {j}' if (j < 4) else 'selection_ID > 3')
    n_sel = tMC.GetEntries(truth_cut + ' && ' + sel_cut)
    print(i, j, n_sel, n_true, n_sel/n_true)
    if i == j and i < 4:
      eff_lines[sets[i]] += f'& \\textbf{{{n_sel/n_true:.2f}}} '
    else:
      eff_lines[sets[i]] += f'& {n_sel/n_true:.2f} '
  eff_lines[sets[i]] += '\\\\'

for o in order:
  print(eff_lines[o])
print()

print('Good Track Efficiencies')
good_track = 'selection_ID != 6 && selection_ID != 5'
for i in range(1, 5):
  truth_cut = (f'new_interaction_topology == {i}' if (i < 4)
               else 'new_interaction_topology > 3')

  n_true = tMC.GetEntries(truth_cut + f' && {good_track}')

  eff_lines[sets[i]] = f'{labels[sets[i]]} '
  for j in range(1, 5):
    sel_cut = (f'selection_ID == {j}' if (j < 4) else 'selection_ID > 3')
    n_sel = tMC.GetEntries(truth_cut + ' && ' + sel_cut + f' && {good_track}')
    print(i, j, n_sel, n_true, n_sel/n_true)
    if i == j and i < 4:
      eff_lines[sets[i]] += f'& \\textbf{{{n_sel/n_true:.2f}}} '
    else:
      eff_lines[sets[i]] += f'& {n_sel/n_true:.2f} '
  eff_lines[sets[i]] += '\\\\'

for o in order:
  print(eff_lines[o])
print()

print('Purities')


for i in range(1, 5):
  sel_cut = (f'selection_ID == {i}' if (i < 4) else 'selection_ID > 3')
  n_sel = tMC.GetEntries(sel_cut)

  line = f'{labels[sets[i]]} '
  for j in range(1, 5):
    truth_cut = (f'new_interaction_topology == {j}' if (j < 4)
                 else 'new_interaction_topology > 3')
    n_true = tMC.GetEntries(truth_cut + ' && ' + sel_cut)
    #print(i, j, n_sel, n_true, n_true/n_sel)
    if i == j and i < 4:
      line += f'& \\textbf{{{n_true/n_sel:.2f}}} '
    else:
      line += f'& {n_true/n_sel:.2f} '
  print(line + '\\\\')

