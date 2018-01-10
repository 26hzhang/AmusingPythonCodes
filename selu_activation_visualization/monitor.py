import subprocess
import argparse
import os.path

parser = argparse.ArgumentParser(description=r'''Launch tensorboard on multiple directories in an easy way.''')
parser.add_argument('--port', default=6006, type=int, help='The port to use for tensorboard')
parser.add_argument('--quiet', '-q', action='store_true', help='Run in silent mode')
parser.add_argument('dirs', nargs='+', type=str, help='directories of train instances to monitor')
args = parser.parse_args()

args.dirs = [s for s in args.dirs if os.path.isdir(s)]
for s in args.dirs:
    print('Monitoring %s ...' % s)
print('')

cmd = 'tensorboard --port="{}" --logdir="{}"'.format(args.port,
                                                     ','.join(["%s:%s" % (os.path.basename(s), s) for s in args.dirs]))
if args.quiet:
    cmd += ' 2>/dev/null'

print(cmd)
subprocess.call(cmd, shell=True)
