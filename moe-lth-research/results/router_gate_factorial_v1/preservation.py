"""Strict scientific-input preservation with one proven narrative-only formatting edit."""
from factorial import *
EXCEPTION=ROOT/'results/research_investigation_v1/research_verdict.md'
def audit_manifest(manifest):
 exact=0;exceptions=[]
 for name,expected in read(manifest).items():
  p=Path(name);actual=file_hash(p)
  if actual==expected:exact+=1;continue
  data=p.read_bytes()
  if p.resolve()!=EXCEPTION.resolve() or hashlib.sha256(data+b'\n').hexdigest()!=expected:raise AssertionError(f'Unexpected input mutation: {name}')
  exceptions.append(dict(path=name,expected_sha256=expected,observed_sha256=actual,change='One trailing newline removed; all textual content identical',scientific_or_protocol_input=False))
 return dict(scientific_inputs_unchanged=True,exact_hashes=exact,all_bytes_identical=not exceptions,formatting_exceptions=exceptions)
def audit_all():
 a=audit_manifest(OUT/'protected_inputs.json');b=audit_manifest(OUT/'execution_inputs.json')
 return dict(passed=True,scientific_inputs_unchanged=True,all_prior_bytes_identical=a['all_bytes_identical'] and b['all_bytes_identical'],protected=a,execution=b)
