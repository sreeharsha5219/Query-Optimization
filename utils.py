import os
import json
import fcntl
import hashlib
import psycopg2
import time

from config import CONNECTION_STR, TIMEOUT, LOG_PATH, SEP


def read_queries(query_path):
    queries = []
    with open(query_path, 'r') as file:
        for line in file.readlines():
            query_name, query = line.strip().split(SEP)
            queries.append((query_name, query))
    return queries


def encode_string(val):
    return hashlib.md5(val.encode('utf-8')).hexdigest()


def explain_query(query, run_args, explain_string):
    query = explain_string + query.strip().replace("\n", " ").replace("\t", " ")
    _, plan_json = execute_query(query, run_args)
    plan_json = plan_json[0][0]

    return [plan_json[1]] if len(plan_json) == 2 else plan_json


def execute_query(query, run_args):
    start_time = time.time()
    result = None
    try:
        with psycopg2.connect(CONNECTION_STR) as conn:
            conn.set_client_encoding('UTF8')
            with conn.cursor() as cur:
                if run_args:
                    for arg in run_args:
                        cur.execute(arg)
                cur.execute("SET statement_timeout TO " + str(TIMEOUT))

                print(query)
                cur.execute(query)
                result = cur.fetchall()
    except Exception as e:
        print(f"[ERROR]: {e}")

    return time.time() - start_time, result


def read_generated_plans(encoded_q_str, plan_str, encoded_plan_str, algo):
    generated_plans_path = os.path.join(f"{LOG_PATH}_{algo}", encoded_q_str, encoded_plan_str)
    if not os.path.exists(generated_plans_path):
        return None

    with open(os.path.join(generated_plans_path, "check_plan"), "r") as file:
        if (generated_plans_str := file.read().strip()) != plan_str:
            print(f"Hash conflict at {generated_plans_path}: given {plan_str}, but found {generated_plans_str}")
            return None

    print(f"getting generated plans from file: {generated_plans_path}")
    with open(os.path.join(generated_plans_path, "plan"), "r") as file:
        return json.loads(file.read().strip())


def write_generated_plans(query, encoded_q_str, plan_str, encoded_plan_str, latency_str, algo):
    generated_plans_path = os.path.join(f"{LOG_PATH}_{algo}", encoded_q_str)
    if not os.path.exists(generated_plans_path):
        os.makedirs(generated_plans_path)
        with open(os.path.join(generated_plans_path, "query"), "w") as file:
            file.write(query)
    else:
        with open(os.path.join(generated_plans_path, "query"), "r") as file:
            if (generated_plans_query := file.read()) != query:
                print(f"Hash conflict at {generated_plans_path}: given {query}, but found {generated_plans_query}")
                return

    generated_plans_path = os.path.join(generated_plans_path, encoded_plan_str)
    if os.path.exists(generated_plans_path):
        print(f"Plan already saved: {generated_plans_path}")
        return

    os.makedirs(generated_plans_path)
    with open(os.path.join(generated_plans_path, "check_plan"), "w") as file:
        file.write(plan_str)
    with open(os.path.join(generated_plans_path, "plan"), "w") as file:
        file.write(latency_str)
    print(f"Generated plan(s) saved at: {generated_plans_path}")


def execute_queries_on_postgres(sql, query_name, run_args, latency_file, write_latency_file=True, algo=""):
    sql = sql.strip().replace("\n", " ").replace("\t", " ")

    plan_json = explain_query(sql, run_args, "EXPLAIN (COSTS FALSE, FORMAT JSON, SUMMARY) ")
    planning_time = plan_json[0]['Planning Time']

    current_plan_str = json.dumps(plan_json[0]['Plan'])
    try:
        encoded_plan_str = encode_string(current_plan_str)
        encoded_query_str = encode_string(sql)
        if (latency_json := read_generated_plans(encoded_query_str, current_plan_str, encoded_plan_str, algo)) is None:
            run_start = time.time()
            try:
                latency_json = explain_query(sql, run_args,
                                             "EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) ")
            except Exception as e:
                if time.time() - run_start > (TIMEOUT / 1000 * 0.9):
                    latency_json = explain_query(sql, run_args,
                                                 "EXPLAIN (VERBOSE, COSTS, FORMAT JSON, SUMMARY) ")
                    latency_json[0]["Execution Time"] = TIMEOUT
                else:
                    raise e

            write_generated_plans(sql, encoded_query_str, current_plan_str, encoded_plan_str, json.dumps(latency_json), algo)

        latency_json[0]['Planning Time'] = planning_time
        if write_latency_file:
            with open(latency_file, "a+") as file:
                fcntl.flock(file, fcntl.LOCK_EX)
                file.write(f"{query_name}{SEP}{json.dumps(latency_json)}\n")
                fcntl.flock(file, fcntl.LOCK_UN)

        print(query_name, latency_json[0]["Execution Time"], flush=True)
        print(query_name + " ----" * 25)
    except Exception as e:
        with open(latency_file + "_error", "a+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(f"{query_name}{SEP}{str(e).strip()}\n")
            fcntl.flock(file, fcntl.LOCK_UN)
