#!/usr/bin/env python3
"""Verify final reported paired results from released seed-level data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
def paired(hard,random):
 d=np.asarray(hard,dtype=float)-np.asarray(random,dtype=float)
 return float(d.mean()),np.asarray(stats.t.interval(.95,len(d)-1,loc=d.mean(),scale=stats.sem(d))),d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['dinov2','ordering','imagenet']);args=ap.parse_args();results={}
 if args.only in [None,'dinov2']:
  package=ROOT/'experiments/dinov2_gate3'
  rows=list(csv.DictReader((package/'statistics/dinov2_clean_5seed_all_conditions.csv').open()))
  expected=json.loads((package/'statistics/dinov2_clean_5seed_all_conditions.json').read_text())['conditions']
  printed={'ordinary_10':-14.87,'ordinary_30':-17.25,'ordinary_50':-11.12,'class_balanced_10':-11.40}
  for name,reference in expected.items():
   sub=[r for r in rows if r['condition']==name];assert sorted(int(r['seed']) for r in sub)==list(range(5))
   mean,ci,d=paired([r['hard_top1'] for r in sub],[r['random_top1'] for r in sub])
   assert np.allclose(d,[float(r['paired_delta_pp']) for r in sub],atol=1e-10)
   assert np.isclose(mean,reference['delta_mean'],atol=1e-10) and np.allclose(ci,reference['ci95'],atol=1e-10) and np.all(d<0)
   assert round(mean,2)==printed[name]
   results[name]={'n':len(d),'delta':mean,'ci95':ci.tolist()}
 if args.only in [None,'ordering']:
  package=ROOT/'experiments/ddpm_ordering';rows=list(csv.DictReader((package/'per_seed_results.csv').open()));assert sorted(int(r['seed']) for r in rows)==list(range(8))
  reference=json.loads((package/'summary.json').read_text())
  for name,a,b in [('easy_minus_random','easy_top1','random_top1'),('shuffled_minus_random','shuffled_top1','random_top1'),('easy_minus_shuffled','easy_top1','shuffled_top1')]:
   mean,ci,d=paired([r[a] for r in rows],[r[b] for r in rows]);assert np.isclose(mean,reference[name]['mean_pp'],atol=1e-10) and np.allclose(ci,reference[name]['ci95_pp'],atol=1e-9)
   results[name]={'n':len(d),'delta':mean,'ci95':ci.tolist()}
 if args.only in [None,'imagenet']:
  package=ROOT/'experiments/imagenet1k_vae_gate3';rows=list(csv.DictReader((package/'SEED_LEVEL_RESULTS.csv').open()));assert sorted(int(r['seed']) for r in rows)==list(range(5))
  reference=json.loads((package/'FINAL_STATISTICS.json').read_text())
  for endpoint,key,printed in [('best','best_checkpoint',-7.765),('final','final_epoch_90',-7.854)]:
   hard=[r['proxy_hard_top1_'+endpoint] for r in rows];random=[r['random_top1_'+endpoint] for r in rows]
   mean,ci,d=paired(hard,random);ref=reference[key]['top1'];assert np.isclose(mean,ref['paired_delta_mean'],atol=1e-10) and np.allclose(ci,ref['paired_95_ci'],atol=1e-10) and np.all(d<0)
   assert round(mean,3)==printed
   if endpoint=='best':assert round(np.mean(np.asarray(hard,dtype=float)),3)==53.646 and round(np.mean(np.asarray(random,dtype=float)),3)==61.412
   results['imagenet_'+endpoint]={'n':len(d),'delta':mean,'ci95':ci.tolist()}
 print(json.dumps({'verification':'PASS','results':results},indent=2))
if __name__=='__main__':main()
