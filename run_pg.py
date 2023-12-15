import os
import argparse
from multiprocessing import Pool

from config import LOG_PATH
from utils import execute_queries_on_postgres, read_queries


class PGExecutor:
    def __init__(self, queries, output_query_latency_file) -> None:
        self.queries = queries
        self.output_query_latency_file = output_query_latency_file

        print("output_query_latency_file:", output_query_latency_file)
        print("Read", len(queries), "queries.")

    def start(self, pool_num):
        print("pool_num:", pool_num)
        pool = Pool(pool_num)
        for fp, q in self.queries:
            pool.apply_async(execute_queries_on_postgres, args=(q, fp, [], self.output_query_latency_file, True, "pg"))
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", metavar="PATH", help="Load the queries")
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--pool_num", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    queries = read_queries(args.query_path)
    queries = queries[:100] if "train" in args.query_path else queries
    PGExecutor(queries, args.output_query_latency_file).start(args.pool_num)
