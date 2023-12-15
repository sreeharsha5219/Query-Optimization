import json
from typing import Union

import numpy as np

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort",
            "Limit"] + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES


def json_str_to_json_obj(json_data) -> dict:
    json_obj = json.loads(json_data)
    return json_obj[0] if isinstance(json_obj, list) and len(json_obj) == 1 else json_obj


class Preprocessor:
    def __init__(self):
        self.normalizer = None
        self.feature_parser = None

    def fit(self, trees):
        def compute_min_max(x):
            x = np.log(np.array(x) + 1)
            return np.min(x), np.max(x)

        exec_times, startup_costs, total_costs, rows, input_relations, rel_types = [], [], [], [], set(), set()

        def parse_tree(n):
            startup_costs.append(n["Startup Cost"])
            total_costs.append(n["Total Cost"])
            rows.append(n["Plan Rows"])
            rel_types.add(n["Node Type"])
            if "Relation Name" in n:
                input_relations.add(n["Relation Name"])

            if "Plans" in n:
                for child in n["Plans"]:
                    parse_tree(child)

        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj["Execution Time"]))
            parse_tree(json_obj["Plan"])

        mins, maxs = {}, {}
        for name, values in [("Startup Cost", startup_costs), ("Total Cost", total_costs),
                             ("Plan Rows", rows), ("Execution Time", exec_times)]:
            if name == "Execution Time" and len(values) == 0:
                continue
            mins[name], maxs[name] = compute_min_max(values)

        self.normalizer = Normalizer(mins, maxs)
        self.feature_parser = AnalyzeJsonParser(self.normalizer, list(input_relations))

    def transform(self, trees):
        local_features = []
        y = []
        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if not isinstance(json_obj["Plan"], dict):
                json_obj["Plan"] = json.loads(json_obj["Plan"])
            local_features.append(self.feature_parser.extract_feature(json_obj["Plan"]))

            if "Execution Time" in json_obj:
                label = float(json_obj["Execution Time"])
                if self.normalizer.contains("Execution Time"):
                    label = self.normalizer.norm(label, "Execution Time")
                y.append(label)
            else:
                y.append(None)
        return local_features, y


class PlanNode:
    def __init__(self, node_type: np.ndarray, startup_cost: Union[float, None], total_cost: Union[float, None],
                 rows: float, width: int, left, right, startup_time: float, total_time: float,
                 input_tables: list, encoded_input_tables: list) -> None:
        self.node_type = node_type
        self.startup_cost = startup_cost
        self.total_cost = total_cost
        self.rows = rows
        self.width = width
        self.left_child = left
        self.right_child = right
        self.startup_time = startup_time
        self.total_time = total_time
        self.input_tables = input_tables
        self.encoded_input_tables = encoded_input_tables

    def get_feature(self):
        return np.hstack((self.node_type, np.array(self.encoded_input_tables), np.array([self.width, self.rows])))

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    def get_subtrees(self):
        trees = [self]
        if self.left_child is not None:
            trees += self.left_child.get_subtrees()
        if self.right_child is not None:
            trees += self.right_child.get_subtrees()
        return trees


class Normalizer:
    def __init__(self, mins: dict, maxs: dict) -> None:
        self.min_vals = mins
        self.max_vals = maxs

    def norm(self, x, name):
        return (np.log(x + 1) - self.min_vals[name]) / (self.max_vals[name] - self.min_vals[name])

    def contains(self, name):
        return name in self.min_vals and name in self.max_vals


class AnalyzeJsonParser:
    def __init__(self, normalizer: Normalizer, input_relations: list) -> None:
        self.normalizer = normalizer
        self.input_relations = input_relations

    def extract_feature(self, json_rel) -> PlanNode:
        left = None
        right = None
        input_relations = []

        if 'Plans' in json_rel:
            children = json_rel['Plans']
            left = self.extract_feature(children[0])
            input_relations += left.input_tables

            if len(children) == 2:
                right = self.extract_feature(children[1])
                input_relations += right.input_tables
            else:
                right = PlanNode(get_one_hot(UNKNOWN_OP_TYPE), 0, 0, 0, 0,
                                 None, None, 0, 0, [], self.encode_relation_names([]))

        node_type = get_one_hot(json_rel['Node Type'])
        rows = self.normalizer.norm(float(json_rel['Plan Rows']), 'Plan Rows')
        width = int(json_rel['Plan Width'])

        if json_rel['Node Type'] in SCAN_TYPES:
            input_relations.append(json_rel["Relation Name"])

        startup_time = None
        if 'Actual Startup Time' in json_rel:
            startup_time = float(json_rel['Actual Startup Time'])
        total_time = None
        if 'Actual Total Time' in json_rel:
            total_time = float(json_rel['Actual Total Time'])

        return PlanNode(node_type, None, None, rows, width, left,
                        right, startup_time, total_time,
                        input_relations, self.encode_relation_names(input_relations))

    def encode_relation_names(self, relation_names):
        encode_arr = np.zeros(len(self.input_relations) + 1)

        for name in relation_names:
            encode_arr[-1 if name not in self.input_relations else list(self.input_relations).index(name)] += 1

        return encode_arr.tolist()


def get_one_hot(op_name):
    arr = np.zeros(len(OP_TYPES))
    arr[OP_TYPES.index(UNKNOWN_OP_TYPE) if op_name not in OP_TYPES else OP_TYPES.index(op_name)] = 1
    return arr
