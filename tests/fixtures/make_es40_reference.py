"""Regenerate vectors with the actual upstream C++ process methods.

Usage: python tests/fixtures/make_es40_reference.py /path/to/vcvrack-encoders
Requires c++ and the pinned checkout. Never imports HydraPy's encoder.
The extracted upstream code remains in a temporary directory.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REVISION = '95496e8a955407889bbab94cf404cf356802bb76'
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', str(root), 'diff', '--', 'src/8GT.cpp', 'src/8CV.cpp', 'src/ES40.cpp'])


def body(file, signature):
    text = (root/'src'/file).read_text()
    start = text.index(signature)
    return text[start:text.index('\nstruct ', start)]


gate_inputs = [[5. if mask & (1 << jack) else 0. for jack in range(8)] for mask in range(256)]
thresholds = [0., .75, 1., .75, .5, .499, .999, 1., 2., -1., .75]
gate_inputs += [[thresholds[(frame + jack) % len(thresholds)] for jack in range(8)]
                for frame in range(2 * len(thresholds))]
gate_inputs += [[0.] * 8]
gate_array = '{' + ','.join('{' + ','.join(f'{v}f' for v in row) + '}' for row in gate_inputs) + '}'

source = '''#include <algorithm>
#include <cstdio>
#include <cstdint>
struct Port {float v=0; float getVoltage(){return v;} void setVoltage(float f){v=f;}};
struct ProcessArgs {};
struct Module8CV {int m_phase=0;unsigned m_value=0;Port inputs[8],outputs[1];void process(const ProcessArgs&);};
struct Module8GT {bool m_bits[8]={};Port inputs[8],outputs[1];void process(const ProcessArgs&);};
struct ModuleES40 {int category=0;Port inputs[5],outputs[2];void process(const ProcessArgs&);};
''' + body('8CV.cpp', 'void Module8CV::process') + body('8GT.cpp', 'void Module8GT::process') + body('ES40.cpp', '#define ES_BITSTOFLOAT') + '''
int main(){
 Module8CV cv; ModuleES40 es[5]; ProcessArgs args;
 for(int i=0;i<101;i++){
  for(int c=0;c<8;c++)cv.inputs[c].v=((i*103+c*719)%7000-3500)/384.0f;
  cv.process(args); printf("%d",(int)cv.outputs[0].v);
  for(int h=0;h<5;h++){
   es[h].inputs[h].v=cv.outputs[0].v; es[h].process(args);
   for(int c=0;c<2;c++)printf(" %d", (int32_t)((double)(es[h].outputs[c].v/10.0f)*8388608));
  }
  printf("\\n");
 }
 Module8GT gt;
 float gates[][8] = ''' + gate_array + ''';
 for(auto& frame:gates){
  for(int c=0;c<8;c++)gt.inputs[c].v=frame[c];
  gt.process(args); printf("GT %d",(int)gt.outputs[0].v);
  for(int h=0;h<5;h++){
   es[h].inputs[h].v=gt.outputs[0].v; es[h].process(args);
   for(int c=0;c<2;c++)printf(" %d", (int32_t)((double)(es[h].outputs[c].v/10.0f)*8388608));
  }
  printf("\\n");
 }
}
'''
with tempfile.TemporaryDirectory() as tmp:
    source_path, binary = Path(tmp)/'reference.cpp', Path(tmp)/'reference'
    source_path.write_text(source)
    subprocess.run(['c++', '-std=c++11', '-O2', str(source_path), '-o', str(binary)], check=True)
    lines = subprocess.check_output([str(binary)], text=True).splitlines()
    rows = [list(map(int, r.split())) for r in lines if not r.startswith('GT ')]
    gate_rows = [list(map(int, r.split()[1:])) for r in lines if r.startswith('GT ')]
fixture = dict(source='expertsleepersltd/vcvrack-encoders@' + REVISION,
               input='float32(((frame*103 + jack*719) % 7000 - 3500) / 384); frames 0..100, jacks 0..7',
               columns='8CV byte, then signed 24-bit L/R words for ES-40 headers 1..5 (category A)', rows=rows)
Path(__file__).with_name('es40-reference.json').write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
gate_fixture = dict(source=fixture['source'], inputs=gate_inputs,
                    columns='8GT byte, then signed 24-bit L/R words for ES-40 headers 1..5 (category A)',
                    rows=gate_rows)
Path(__file__).with_name('es40-gates-reference.json').write_text(json.dumps(gate_fixture, separators=(',', ':')) + '\n')
