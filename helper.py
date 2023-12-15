from model import LeroModelPairWise
from pre_process import Preprocessor


def load_pairwise_plans(path):
    x1, x2 = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            a, b = get_training_pair(arr)
            x1.extend(a)
            x2.extend(b)

    return x1, x2


def get_training_pair(candidates):
    x1, x2 = [], []

    for i in range(len(candidates) - 1):
        s1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            s2 = candidates[j]
            x1.append(s1)
            x2.append(s2)

    return x1, x2


def training_pairwise(model_name, training_data_file):
    x1, x2 = load_pairwise_plans(training_data_file)

    feature_generator = Preprocessor()
    feature_generator.fit(x1 + x2)

    x1, y1 = feature_generator.transform(x1)
    x2, y2 = feature_generator.transform(x2)
    print("Training data set size = " + str(len(x1)))

    lero_model = LeroModelPairWise(feature_generator)
    lero_model.fit(x1, x2, y1, y2)

    print("saving model...")
    lero_model.save(model_name)


def run(training_data, model_name):
    print("training_data:", training_data)
    print("model_name:", model_name)

    training_pairwise(model_name, training_data)
