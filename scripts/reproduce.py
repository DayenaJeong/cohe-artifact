#!/usr/bin/env python3
"""Verified zero-GPU result/statistic entry points; never launches training."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
COMMANDS={
 'dependence':['scripts/run_stage1_dependence.py','--proxy','derived_scalar_arrays/cifar100_ddim_proxy_scores.csv','--target','derived_scalar_arrays/cifar100_ce_targets.csv','--out','outputs/ddim_ce_dependence.csv'],
 'predictive_validity':['scripts/run_stage2_prediction.py','--proxy','derived_scalar_arrays/cifar100_dinov2_proxy_scores.csv','--target','derived_scalar_arrays/cifar100_first_learning_targets.csv','--splits','derived_scalar_arrays/cifar100_split_ids.csv','--out','outputs/dinov2_linear_interface_prediction.csv'],
 'dinov2_gate3':['reproduction/verify_paper_results.py','--only','dinov2'],
 'ddpm_ordering':['experiments/ddpm_ordering/reproduce_statistics.py'],
 'imagenet1k_vae_gate3':['experiments/imagenet1k_vae_gate3/reproduction/verify_statistics.py'],
 'paper_results':['reproduction/verify_paper_results.py'],
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',choices=COMMANDS,required=True);args=p.parse_args()
 subprocess.run([sys.executable,*COMMANDS[args.experiment]],cwd=ROOT,check=True)
if __name__=='__main__':main()
