import csv
import os
import time

from ontomap.ontology import ontology_matching
from ontomap.base import BaseConfig
from ontomap.evaluation.evaluator import evaluator
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval
from ontomap.postprocess import process

RESULTS_PATH = "my_test_result/digitalhum_results.csv"
MODEL_NAME = "BERTRetrieval"

# Setting configurations for experimenting 'retrieval' on CPU
config = BaseConfig(approach="retrieval").get_args(device="cpu")
# set dataset directory
config.root_dir = "datasets"

write_header = not os.path.exists(RESULTS_PATH)
results_file = open(RESULTS_PATH, "a", newline="")
results_writer = csv.writer(results_file)
if write_header:
    results_writer.writerow(["dataset", "model", "time(s)", "result"])

# run every sub-dataset registered under the "digitalhum" track
for dataset_cls in ontology_matching["digitalhum"]:
    dataset = dataset_cls()
    print(f"\n======== {dataset.ontology_name} ========")
    start_time = time.time()

    # parse task source, target, and reference ontology
    ontology = dataset.collect(root_dir=config.root_dir)

    # init encoder (concept-representation)
    encoded_inputs = IRILabelInLightweightEncoder()(
        source=ontology["source"], target=ontology["target"]
    )

    # init BERTRetrieval
    model = BERTRetrieval(
        top_k=config.BERTRetrieval["top_k"], device=config.BERTRetrieval["device"]
    )
    # generate results
    predicts = model.generate(input_data=encoded_inputs)

    # post-processing
    predicts = process.eval_preprocess_ir_outputs(predicts=predicts)
    predicts = list(filter(lambda x: x["score"] > 0.9, predicts))

    # evaluation
    results = evaluator(
        track="digitalhum", predicts=predicts, references=ontology["reference"]
    )
    elapsed = round(time.time() - start_time, 2)
    print(f"{dataset.ontology_name}: {results}")

    results_writer.writerow([dataset.ontology_name, MODEL_NAME, elapsed, results])
    results_file.flush()

results_file.close()
print(f"\nResults written to {RESULTS_PATH}")
