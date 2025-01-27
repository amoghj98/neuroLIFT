#!/usr/bin/python

import argparse
from argparse import RawTextHelpFormatter


def config():
	desc=("NeuroLIFT Finetuning Framework arg parser")
	epilog=("For further documentation, refer the NeuroLIFT framework documentation page at https://github.com/")
	parser=argparse.ArgumentParser(description=desc, epilog=epilog, formatter_class=RawTextHelpFormatter)
	# policy training args
	parser.add_argument('-m', '--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Name of LLM to finetune')
	parser.add_argument('-l', '--nLabels', type=int, default=2, help='Number of output classifier labels')
	parser.add_argument('-mp', '--modelPath', type=str, default='/scratch/gautschi/joshi157/models/', help='Absolute path of the irectory containing model weights')
	# parse all args
	args = parser.parse_args()
	return args