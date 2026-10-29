import json,re,os,sys
import rwkv
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER(os.path.join(os.path.dirname(rwkv.__file__),'rwkv_vocab_v20230424.txt'))
strict=re.compile(r'teacher note|at least one search|as short as possible|private note',re.I)
loose=re.compile(r'(teacher note|at least one search|as short as possible|keep (your|my) thinking brief|no markdown|system prompt|instructions? (say|says|state)|the prompt (says|requires)|private note|not (to )?mention)',re.I)
def pct(a,q): a=sorted(a); return a[min(len(a)-1,int(len(a)*q))]
for name in sys.argv[1:]:
    T=[json.loads(l) for l in open(f'outputs/distill/{name}/trajectories.jsonl')]
    ok=[t for t in T if t.get('steps')]
    C=[t for t in ok if t.get('status')=='answer' and t.get('correct') and t.get('n_search',0)>=1]
    st=[s for t in ok for s in t['steps']]
    L=[len(tok.encode(s['reasoning'])) for s in st]
    print(f"{name}: EM {sum(t.get('correct',False) for t in ok)/len(ok):.3f} | 全部步 {len(st)}: 明确复述 {sum(bool(strict.search(s['reasoning'])) for s in st)/len(st):.1%}, 宽口径 {sum(bool(loose.search(s['reasoning'])) for s in st)/len(st):.1%} | 答对轨迹 {len(C)} 条中含复述 {sum(any(loose.search(s['reasoning']) for s in t['steps']) for t in C)/max(len(C),1):.1%} | 思考 token p50 {pct(L,.5)} p90 {pct(L,.9)}")
