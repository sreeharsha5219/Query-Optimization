import os
import json
import socket
import warnings
import argparse
from multiprocessing import Pool
from collections import defaultdict

from helper import run
from utils import execute_queries_on_postgres, read_queries
from config import SEP, LOG_PATH, LERO_SERVER_PORT, LERO_SERVER_HOST, LERO_SERVER_PATH, LERO_DUMP_CARD_FILE, PG_DB_PATH


warnings.filterwarnings("ignore")


class CardinalityGuidedEntity:
    def __init__(self, score, card_str) -> None:
        self.score = score
        self.card_str = card_str

    def get_score(self):
        return self.score


def create_training_file(training_data_file, *latency_files):
    pair_dict = defaultdict(lambda: [])

    for latency_file in latency_files:
        with open(latency_file, 'r') as file:
            for _line in file.readlines():
                key, value = _line.strip().split(SEP)
                pair_dict[key].append(value)

    training_data = [SEP.join(values) for values in pair_dict.values() if len(values) > 1]

    with open(training_data_file, 'w') as f2:
        f2.write("\n".join(training_data))


def generate_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def load_model(model_name):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        json_str = json.dumps({"msg_type": "load", "model_path": os.path.abspath(LERO_SERVER_PATH + model_name)})
        print("load_model", json_str)

        s.sendall(bytes(f"{json_str}*LERO_END*", "utf-8"))
        reply_json = s.recv(1024)

    print(reply_json.decode("utf-8"))
    os.system("sync")


class LeroExecutor:
    def __init__(self, queries, query_num_per_chunk, output_query_latency_file,
                 test_queries, model_prefix, top_k) -> None:
        self.queries = queries
        self.query_num_per_chunk = query_num_per_chunk
        self.output_query_latency_file = output_query_latency_file
        self.test_queries = test_queries
        self.model_prefix = model_prefix
        self.top_k = top_k
        self.lero_server_path = LERO_SERVER_PATH
        self.lero_card_file_path = os.path.join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)

        print("Read", len(queries), "training queries.")
        print("Read", len(test_queries), "test queries.")

        print("topK:", top_k)
        print("model_prefix:", model_prefix)
        print("query_num_per_chunk:", query_num_per_chunk)
        print("output_query_latency_file:", output_query_latency_file)

    def start(self, pool_num):
        print("pool_num: ", pool_num)
        lero_chunks = list(generate_chunks(self.queries, self.query_num_per_chunk))
        for c_idx, chunk in enumerate(lero_chunks):
            with Pool(pool_num) as pool:
                for query_name, query in chunk:
                    self.run_pairwise(query, query_name, pool)
                pool.close()
                pool.join()

            model_name = f"model/{self.model_prefix}_{c_idx}"
            self.retrain(model_name)
            self.test(f"{self.output_query_latency_file}_{model_name}")

            if c_idx == 4:
                break

    def retrain(self, model_name):
        training_data_file = f"{self.output_query_latency_file}.training"
        create_training_file(training_data_file, self.output_query_latency_file,
                             f"{self.output_query_latency_file}_exploratory")
        print(f"Retrain Lero model: {model_name} with file {training_data_file}")
        run(os.path.abspath(training_data_file), model_name)
        load_model(model_name)

    def test(self, output_file):
        run_args = ["SET enable_lero TO True"]
        for query_name, query in self.test_queries:
            execute_queries_on_postgres(query, query_name, run_args, output_file, True, "pg")

    def run_pairwise(self, query, query_name, pool):
        entities = []
        with open(self.lero_card_file_path, 'r') as file:
            for _line in file.readlines():
                card_str, score = _line.strip().split(";")
                entities.append(CardinalityGuidedEntity(float(score), card_str))

        entities = sorted(entities, key=lambda x: x.get_score())

        for i, entity in enumerate(entities[: self.top_k]):
            if isinstance(entity, CardinalityGuidedEntity):
                card_file_name = f"lero_{query_name}_{i}.txt"
                with open(os.path.join(PG_DB_PATH, card_file_name), "w") as card_file:
                    card_file.write("\n".join(entity.card_str.strip().split(" ")))

                output_file = self.output_query_latency_file if i == 0 else f"{self.output_query_latency_file}_exploratory"
                pool.apply_async(execute_queries_on_postgres,
                                 args=(query, query_name, ["SET lero_joinest_fname TO '" + card_file_name + "'"],
                                       output_file, True, "pg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", metavar="PATH", help="Load the queries")
    parser.add_argument("--test_query_path", metavar="PATH", help="Load the test queries")
    parser.add_argument("--query_num_per_chunk", type=int)
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--pool_num", type=int, default=10)
    parser.add_argument("--topK", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    LeroExecutor(read_queries(args.query_path), args.query_num_per_chunk, args.output_query_latency_file,
                 read_queries(args.test_query_path), args.model_prefix, args.topK).start(args.pool_num)
