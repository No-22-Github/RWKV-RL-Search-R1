"""从 Search-R1 test 集中按 data_source 各抽 N 道题，用于基模能力摸底。"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/search_r1/test.parquet")
    parser.add_argument("--output", default="data/search_r1/test_sample.parquet")
    parser.add_argument("--sources", nargs="+", default=["nq", "hotpotqa"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    parts = [df[df.data_source == s].sample(n=args.n, random_state=args.seed) for s in args.sources]
    out = pd.concat(parts).reset_index(drop=True)
    out.to_parquet(args.output)
    print(out.data_source.value_counts().to_string())
    print(f"saved {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
