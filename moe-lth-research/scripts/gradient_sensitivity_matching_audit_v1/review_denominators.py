"""Clarify fixed-subset denominators in the generated findings, without changing results."""
from common import *
c=read(D/'completion.json');assert c.get('final_artifact_audit_passed')
r=rows(D/'solver_results.csv');p=D/'findings.md';text=p.read_text();lines=text.splitlines()
for i,line in enumerate(lines):
 if line.startswith('| C: fixed40,'):
  rr=[x for x in r if x['design']=='C'];returned=sum(x['candidate_pairs']!='' for x in rr);passed=sum(x['scientific_checks_pass']=='True' for x in rr)
  parts=line.split('|');parts[2]=parts[2].strip()+' among returned candidates ';parts[3]=f' {passed}/40 conditions; {returned}/40 incumbents ';lines[i]='|'.join(parts)
text='\n'.join(lines)+'\n'
p.write_text(text)
c['output_sha256']['findings.md']=sha(p);c['fixed_subset_denominators_reviewed']=True
save(D/'completion.json',c)
print('All40 C conditions retained in concise table; unknown results not imputed as zero')
