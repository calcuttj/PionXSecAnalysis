import ROOT as RT
from argparse import ArgumentParser as ap
import time

def get_events(args):
    f = RT.TFile.Open(args.i)
    t = f.Get('pduneana/beamana')

    with open(args.o, 'w') as fout:

        for i,e in enumerate(t):
            if not i%1000: print(i, end='\r')
            if args.n > 0 and i >= args.n: break
            if e.primary_isBeamType: continue

            fout.write(f'{e.run}:{e.subrun}:{e.event}\n')

    f.Close()

def get_files(args):
    import samweb_cli
    samweb = samweb_cli.SAMWebClient(experiment='dune')

    files = {}
    events = {}
    with open(args.i, 'r') as f:
        lines = [i.strip() for i in f.readlines()]

    with open(args.o, 'w') as fout:
        for i, l in enumerate(lines):
            if args.n > 0 and i >= args.n: break
            run, subrun, event = l.split(':')
            print(l)
            if run not in files.keys():
                print('querying')
                query = f'run_number {run} and run_type protodune-sp and data_tier raw'
                files[run] = samweb.listFiles(query)
                print(f'Got {len(files[run])}')

            thefiles = files[run]
            for f in thefiles:
                print('Searching', f)
                if f not in events.keys():
                    print('Getting events for', f)
                    events[f] = samweb.getURL('/files/metadata/event_numbers', {'file_name': f}).json()
                    print(f'\tGot {len(events[f])}')
                theevents = events[f]
                if int(event) in theevents:
                    print(f'Found {run} {event} in {f}')
                    fout.write(f'{l}:{f}:{theevents.index(int(event))}\n')
                    # time.sleep(1)
                    break

if __name__ == '__main__':
    parser = ap()
    parser.add_argument('routine', type=str, choices=['get_events', 'get_files'])
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-n', type=int, default=-1)
    args = parser.parse_args()

    routines = {
        'get_events':get_events,
        'get_files':get_files,
    }

    routines[args.routine](args)