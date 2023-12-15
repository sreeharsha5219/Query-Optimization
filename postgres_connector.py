import math
import json
import configparser
import socketserver
from collections import defaultdict

from model import LeroModelPairWise
from pre_process import SCAN_TYPES, JOIN_TYPES


class PostgresHandler(socketserver.BaseRequestHandler):
    def handle(self):
        str_buf = ""
        while True:
            str_buf += self.request.recv(81960).decode("UTF-8")
            if not str_buf:
                return

            end_loc = str_buf.find("*LERO_END*")
            if end_loc != -1:
                json_msg = str_buf[:end_loc].strip()
                str_buf = str_buf[end_loc + len("*LERO_END*"):]
                if json_msg:
                    # try:
                    json_obj = json.loads(json_msg)
                    msg_type = json_obj['msg_type']
                    reply_msg = {}
                    handler_method = getattr(self, f"_handle_{msg_type}", None)
                    if handler_method:
                        handler_method(json_obj, reply_msg)
                        reply_msg['msg_type'] = "succ"
                    else:
                        raise Exception(f"[{msg_type}] not known")

                    # except Exception as e:
                    #     print(json.loads(json_msg))
                    #     reply_msg = {'msg_type': "error", 'error': str(e)}
                    #     print(str(e))

                    self.request.sendall(bytes(json.dumps(reply_msg), "utf-8"))
                    self.request.close()

                    break

    def _handle_init(self, json_obj, _):
        self.server.opt_state_dict[json_obj['query_id']] = OptState(
            Selector(json_obj['rows_array'], json_obj['table_array']),
            PlanSelector(json_obj['rows_array'], json_obj['table_array'])
        )

    def _handle_guided_optimization(self, json_obj, reply_msg):
        opt_state = self.server.opt_state_dict[json_obj['query_id']]
        opt_state.plan_card_replacer.process_plan(json_obj['Plan'])
        new_json_msg = json.dumps(json_obj)

        self._handle_predict(new_json_msg, reply_msg)

        signature = str(get_tree_signature(json_obj['Plan']['Plans'][0]))
        if signature not in opt_state.visited_trees:
            card_list = opt_state.card_picker.get_card_list()
            opt_state.card_list_with_score.append(([str(card) for card in card_list], reply_msg['latency']))
            opt_state.visited_trees.add(signature)

        finish = opt_state.card_picker.next()
        reply_msg['finish'] = 1 if finish else 0

    def _handle_predict(self, json_obj, reply_msg):
        local_features, _ = self.server.feature_generator.transform([json_obj])
        reply_msg['latency'] = self.server.model(local_features)[0][0]

    def _handle_join_card(self, json_obj, reply_msg):
        reply_msg['join_card'] = self.server.opt_state_dict[json_obj['query_id']].card_picker.get_card_list()

    def _handle_load(self, json_obj, _):
        model_path = json_obj['model_path']

        lero_model = LeroModelPairWise(None, model_path=model_path)

        self.server.model = lero_model
        self.server.feature_generator = lero_model.feature_generator

    def _handle_reset(self, _, __):
        self.server.model = None
        self.server.feature_generator = None

    def _handle_remove_state(self, json_obj, _):
        del self.server.opt_state_dict[json_obj['query_id']]


def generate_swing_factors(swing_factor_lower_bound: float = 0.01,
                           swing_factor_upper_bound: int = 100, step: float = 10):
    swing_factors = {1.0}

    num_steps_up = math.ceil(math.log(swing_factor_upper_bound, step))
    swing_factors.update(step ** i for i in range(num_steps_up + 1))

    cur_swing_factor = 1 / step
    while cur_swing_factor >= swing_factor_lower_bound:
        swing_factors.add(cur_swing_factor)
        cur_swing_factor /= step

    swing_factors.add(swing_factor_upper_bound)
    swing_factors.add(swing_factor_lower_bound)

    return sorted(list(swing_factors))


def create_table_num_to_card_index_dict(table_arr):
    table_num_to_card_index, max_table_num = defaultdict(lambda: []), -1
    for i, tables in enumerate(table_arr):
        table_num = len(tables)
        table_num_to_card_index[table_num].append(i)
        max_table_num = max(max_table_num, table_num)

    return table_num_to_card_index, max_table_num


def get_tree_signature(json_tree):
    signature = {}
    node_type = json_tree['Node Type']

    if "Plans" in json_tree:
        children = json_tree['Plans']
        assert len(children) in [1, 2], "Number of children must be 1 or 2"

        signature['L'] = get_tree_signature(children[0])
        if len(children) == 2:
            signature['R'] = get_tree_signature(children[1])

    if node_type in SCAN_TYPES:
        signature["T"] = json_tree['Relation Name']
    elif node_type in JOIN_TYPES:
        signature["J"] = node_type[0]

    return signature


class Selector:
    def __init__(self, rows_arr, table_arr):
        self.rows_arr = rows_arr

        self.table_num_2_card_idx_dict, self.max_table_num = create_table_num_to_card_index_dict(table_arr)
        self.swing_factors = generate_swing_factors()

        self.cur_sub_query_table_num = self.max_table_num
        self.cur_sub_query_related_card_idx_list = self.table_num_2_card_idx_dict[self.cur_sub_query_table_num]

        self.sub_query_swing_factor_index = 0
        self.finish = False

    def get_card_list(self):
        if len(self.cur_sub_query_related_card_idx_list) == 0:
            self.cur_sub_query_related_card_idx_list = self.table_num_2_card_idx_dict[self.cur_sub_query_table_num]

        return [float(item) * (self.swing_factors[
                                   self.sub_query_swing_factor_index] if idx in self.cur_sub_query_related_card_idx_list else 1)
                for idx, item in enumerate(self.rows_arr)]

    def next(self):
        self.sub_query_swing_factor_index += 1
        if self.sub_query_swing_factor_index >= len(self.swing_factors):
            self.sub_query_swing_factor_index = 0
            self.cur_sub_query_table_num -= 1
            self.cur_sub_query_related_card_idx_list = []

        if self.cur_sub_query_table_num <= 1:
            self.cur_sub_query_table_num = self.max_table_num
            self.finish = True

        return self.finish


class PlanSelector:
    def __init__(self, rows_array, table_array) -> None:
        self.table_array = table_array
        self.rows_array = rows_array
        self.SCAN_TYPES = SCAN_TYPES
        self.JOIN_TYPES = JOIN_TYPES
        self.SAME_CARD_TYPES = ["Hash", "Materialize", "Sort", "Incremental Sort", "Limit"]
        self.OP_TYPES = ["Aggregate", "Bitmap Index Scan"] + self.SCAN_TYPES + self.JOIN_TYPES + self.SAME_CARD_TYPES

        self.table_idx_map = {}
        for arr in table_array:
            for t in arr:
                if t not in self.table_idx_map:
                    self.table_idx_map[t] = len(self.table_idx_map)

        self.table_num = len(self.table_idx_map)
        self.table_card_map = {}
        for i in range(len(table_array)):
            arr = table_array[i]
            card = rows_array[i]
            code = self.encode_input_tables(arr)
            if code not in self.table_card_map:
                self.table_card_map[code] = card

    def process_plan(self, plan):
        input_card, input_tables, output_card = None, [], None

        if "Plans" in plan:
            children = plan['Plans']
            for child in children:
                if 'Node Type' not in child:
                    child['Node Type'] = plan['Node Type']
                child_input_card, child_input_tables = self.process_plan(child)
                input_card = child_input_card if len(plan['Plans']) == 1 else input_card
                input_tables.extend(child_input_tables)

        if plan['Node Type'] in self.JOIN_TYPES:
            tag = self.encode_input_tables(input_tables)
            card = self.table_card_map.get(tag, 0)
            plan['Plan Rows'] = card
            output_card = card
        elif plan['Node Type'] in self.SAME_CARD_TYPES:
            if input_card is not None:
                plan['Plan Rows'] = input_card
                output_card = input_card
        elif plan['Node Type'] in self.SCAN_TYPES:
            input_tables.append(plan['Relation Name'])
        elif plan['Node Type'] not in self.OP_TYPES:
            raise Exception(f"[{plan['Node Type']}] not known")

        return output_card, input_tables

    def encode_input_tables(self, input_table_list):
        counts = [0] * self.table_num
        for input_table in input_table_list:
            idx = self.table_idx_map.get(input_table)
            if idx is not None:
                counts[idx] += 1

        return sum(count * (10 ** idx) for idx, count in enumerate(counts))


class OptState:
    def __init__(self, card_picker, plan_card_replacer):
        self.card_picker = card_picker
        self.plan_card_replacer = plan_card_replacer
        self.card_list_with_score = []
        self.visited_trees = set()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("connector.conf")
    config = config["lero"]

    port = int(config["port"])
    listen_on = str(config["host"])

    model = LeroModelPairWise(None, model_path=str(config["modelpath"])) if "modelpath" in config else None
    print("Loaded model from ", config["modelpath"] if "modelpath" in config else None)

    print(f"Listening on {listen_on} port {port}")

    with socketserver.TCPServer((listen_on, port), PostgresHandler) as server:
        server.model = model
        server.feature_generator = model.feature_generator if "modelpath" in config else None
        server.opt_state_dict = {}

        server.best_plan = None
        server.best_score = None

        server.serve_forever()
