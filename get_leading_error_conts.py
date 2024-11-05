import ROOT as RT
from argparse import ArgumentParser as ap
import numpy

def good_name(args, i):
  return (('stat' not in i) and ('PostFit' in i) and ('_' in i) and
          ((args.frac and 'frac' in i) or ((not args.frac) and ('frac' not in i))))

def get_syst_name(args, k):
  #if ('multi' in k or 'costheta' in k or 'momentum' in k or 'p_cos' in k):
  index = -1 if not args.frac else -2
  if ('costheta' in k or 'momentum' in k or 'p_cos' in k):
    if args.frac:
      return '_'.join(k.split('_')[-3:-1])
    else:
      return '_'.join(k.split('_')[-2:])
  else: return k.split('_')[index]
  
if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', type=str)
  parser.add_argument('--frac', action='store_true')
  args = parser.parse_args()

  f = RT.TFile.Open(args.i)

  hs = {i.GetName():f.Get(i.GetName()) for i in f.GetListOfKeys() if good_name(args, i.GetName())}
  print(hs.keys())

  names = ['Abs', 'Cex', 'Other']

  errs = {n:[[] for i in range(3)] for n in names}
  for n in names:
    for k in hs.keys():
      if n in k:
        syst_name = get_syst_name(args, k)
        print(k, syst_name)
        if syst_name == '': print('here')
        h = hs[k]
        for i in range(1, h.GetNbinsX()+1):
          errs[n][i-1].append((numpy.sqrt(h.GetBinContent(i)), syst_name))

  #0 [(6.062185631257056, 'p_multiplicity'), (6.058341545886376, 'sce'), (5.644349769379903, 'n_multiplicity'), (5.253908573441651, 'p_costheta'), (5.246747142137508, 'cal')]
  #1 [(6.571711725594176, 'p_multiplicity'), (6.180993917163922, 'n_multiplicity'), (4.441754488148386, 'sce'), (4.326736728701293, 'p_costheta'), (3.8529925823266504, 'p_momentum')]
  #2 [(5.5375036768640005, 'p_multiplicity'), (5.282981093992475, 'n_multiplicity'), (4.090070849146889, 'p_costheta'), (3.519284463284746, 'p_momentum'), (3.2754560237421346, 'pilow')]
  #0 [(3.831699827523407, 'p_momentum'), (3.31791738295909, 'p_multiplicity'), (2.5938362092957647, 'psigma'), (2.511003095516271, 'cal'), (2.4659263439227557, 'pi0_momentum')]
  #1 [(3.9012076063563788, 'p_costheta'), (3.6972643714388176, 'p_momentum'), (3.0508372313520793, 'p_multiplicity'), (2.476940003050618, 'pi0_multiplicity'), (2.420903307803011, 'psigma')]
  #2 [(5.561673111709227, 'sce'), (4.889703675312303, 'p_momentum'), (3.581857128857268, 'p_costheta'), (2.7312451235955955, 'pi0_multiplicity'), (2.429156121232269, 'cal')]
  #0 [(7.282386348057323, 'sce'), (6.384887751876211, 'p_costheta'), (6.03804192765692, 'cal'), (5.9827030181746546, 'p_momentum'), (4.7993029498078075, 'pilow')]
  #1 [(5.8837134218112235, 'sce'), (5.079333904876102, 'cal'), (4.6017735112167415, 'p_momentum'), (4.556499100311789, 'p_costheta'), (4.287470499647511, 'pihigh')]
  #2 [(5.509377330815476, 'cal'), (5.415962948336766, 'sce'), (4.7512980972693795, 'pihigh'), (4.028371848722273, 'psigma'), (3.7129681059879753, 'piminus_multiplicity')]

  titles = {
    'Abs':'Abs. Bin',
    'Cex':'Ch. Exch. Bin',
    'Other':'Other Bin',
    'pilow':'$\sigma^{\\textrm{Low}}_{\pi^{{+}}}$',
    'pihigh':'$\sigma^{\\textrm{High}}_{\pi^{{+}}}$',
    'psigma':'$\sigma_p$',
    'n_multiplicity':'N$n$',
    'p_multiplicity':'N$p$',
    'pi0_multiplicity':'N$\pi^{{0}}$',
    'piplus_multiplicity':'N$\pi^{{+}}$',
    'piminus_multiplicity':'N$\pi^{{-}}$',
    'p_costheta':'$\\angle_p$',
    'p_momentum':'$p_p$',
    'piplus_costheta':'$\\angle_{\pi^{{+}}}$',
    'piplus_momentum':'$p_{\pi^{+}}$',
    'pi0_costheta':'$\\angle_{\pi^{{0}}}$ ',
    'pi0_momentum':'$p_{\pi^{{0}}}$',
    'sce':'SCE',
    'cal':'Cal.',
    #'p_cos':'FS $p-\\cos\\theta$',
    'p_cos':'FS Kin.',
    'multi':'FS Mult.',
    'scrapers':'Beam Scrapers',
    'fit':'Iter. Fit',
  }

  #titles = {
  #  'Abs':'Absorption Bin',
  #  'Cex':'Ch. Exch. Bin',
  #  'Other':'Other Bin',
  #  'pilow':'$\pi^{{+}}$ $\sigma$ Low',
  #  'pihigh':'$\pi^{{+}}$ $\sigma$ High',
  #  'psigma':'$p$ $\sigma$',
  #  'n_multiplicity':'N Mult.',
  #  'p_multiplicity':'P Mult.',
  #  'pi0_multiplicity':'$\pi^{{0}}$ Mult.',
  #  'piplus_multiplicity':'$\pi^{{+}}$ Mult.',
  #  'piminus_multiplicity':'$\pi^{{-}}$ Mult.',
  #  'p_costheta':'P Angle',
  #  'p_momentum':'P Mom.',
  #  'piplus_costheta':'$\pi^{{+}}$ Angle',
  #  'piplus_momentum':'$\pi^{{+}}$ Mom.',
  #  'pi0_costheta':'$\pi^{{0}}$ Angle',
  #  'pi0_momentum':'$\pi^{{0}}$ Mom.',
  #  'sce':'SCE',
  #  'cal':'Cal.',
  #}

  for n, errvals in errs.items():
    for i in range(len(errvals)):
      vals = errvals[i]
      vals.sort(key=lambda v: v[0], reverse=True)
      #print(n)
      line = f'{titles[n]} {i} '
      for j in range(5):
        #print(vals[j][1])
        #line += f'& {titles[vals[j][1]]} ({vals[j][0]:.2f}) '
        line += f'& {titles[vals[j][1]]} & {vals[j][0]:.2f} '
      line += '\\\\'
      print(line)
      #print(i, vals[:5])

  f.Close()
