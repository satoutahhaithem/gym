#!/usr/bin/env python
import argparse
from dataset.dataset import get_dataset

def main():
  parser = argparse.ArgumentParser(description="Download dataset files for DistributedSim")
  parser.add_argument("--dataset", type=str, default="owt",
                      help="Dataset identifier (default: owt)")
  parser.add_argument("--proportion", type=float, default=1.0,
                      help="Proportion of dataset to download (0.0-1.0)")
  
  args = parser.parse_args()
  
  # Call get_dataset which will download and cache the data
  # We discard the returned dataset since we only want to download
  _, _ = get_dataset(args.dataset, 0, args.proportion)
  
  print("Dataset download complete!")

if __name__ == "__main__":
  main() 