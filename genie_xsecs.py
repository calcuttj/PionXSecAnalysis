from array import array
import ROOT as RT
import numpy as np
from argparse import ArgumentParser as ap

def process(in_name, out_name, nevents):
    f = RT.TFile.Open(in_name)
    t = f.Get('ginuke')



    fout = RT.TFile(out_name, 'recreate')
    tout = RT.TTree('tree', '')
    ke_out = array('d', [0])
    npic_out = array('i', [0])    
    npi0_out = array('i', [0])
    int_type = array('i', [0])

    tout.Branch('ke', ke_out, 'ke/D')
    tout.Branch('npic', npic_out, 'npic/I')
    tout.Branch('npi0', npi0_out, 'npi0/I')
    tout.Branch('int_type', int_type, 'int_type/I')
    for i, e in enumerate(t):
        if not i % 1000: print(f'{i}/{t.GetEntries()}', end='\r')
        if nevents > 0 and i >= nevents: break
        ke_out[0] = e.ke
        if e.probe_fsi == 1:
            int_type[0] = 0
            tout.Fill()
            continue

        pdgs = np.array(e.pdgh)
        pi0_indices = np.where(pdgs == 111)
        pic_indices = np.where(abs(pdgs) == 211)
        momenta = np.array(e.ph)
        # print(momenta)
        # print(pdgs)

        above_thresh_pic = np.intersect1d(
            np.where(momenta > .150),
            pic_indices
        )

        n_thresh_pic = len(above_thresh_pic)
        n_pi0 = len(pi0_indices[0])
        
        # print(n_thresh_pic, n_pi0)

        npi0_out[0] = n_pi0
        npic_out[0] = n_thresh_pic

        if n_thresh_pic == 0 and n_pi0 == 0:
            int_type[0] = 1
        elif n_thresh_pic == 0 and n_pi0 > 0:
            int_type[0] = 2
        else:
            int_type[0] = 3
        tout.Fill()
        

    f.Close()
    fout.cd()
    tout.Write()
    fout.Close()

def draw(in_name, out_name):
    f = RT.TFile.Open(in_name)
    t = f.Get('tree')

    fout = RT.TFile(out_name, 'recreate')
    t.Draw('1.e3*ke>>hD(120, 0, 1200)')
    hD = RT.gDirectory.Get('hD')

    int_type = {
        'abs':1,
        'cex':2,
        'other':3,
    }

    hs = []
    A = 40
    r0 = 3*1.4
    r = r0*(A**(1./3.))
    scale_factor = 10*RT.TMath.Pi() * r**2 
    for n, it in int_type.items():
        print('Doing', n)
        t.Draw(f'1.e3*ke>>h{n}(120, 0, 1200)', f'int_type=={it}')
        hs.append(RT.gDirectory.Get(f'h{n}'))
        hs[-1].Divide(hD)
        hs[-1].Scale(scale_factor)

    t.Draw('ke>>hTotal(120, 0, 1.2)', 'int_type>0')
    hs.append(RT.gDirectory.Get(f'hTotal'))
    hs[-1].Divide(hD)
    hs[-1].Scale(scale_factor)

    f.Close()
    fout.cd()
    for h in hs: fout.Write()


if __name__ == '__main__':
    parser = ap()
    parser.add_argument('routine', type=str, choices=['process', 'draw'])
    parser.add_argument('-i', type=str, default=None)
    parser.add_argument('-o', type=str, default=None)
    parser.add_argument('-n', type=int, default=-1)

    args = parser.parse_args()

    if args.routine == 'process':
        process(args.i, args.o, args.n)
    elif args.routine == 'draw':
        draw(args.i, args.o)
    