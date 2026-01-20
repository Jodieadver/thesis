from ontomap.ontology import MouseHumanOMDataset
from ontomap.base import BaseConfig
from ontomap.evaluation.evaluator import evaluator
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval
from ontomap.postprocess import process
import subprocess

#  original code
# Setting configurations for experimenting 'retrieval' on CPU
config = BaseConfig(approach="retrieval").get_args(device="cpu")
# set dataset directory
config.root_dir = "datasets"
# parse task source, target, and reference ontology
ontology = MouseHumanOMDataset().collect(root_dir=config.root_dir)

# init encoder (concept-representation)
encoded_inputs = IRILabelInLightweightEncoder()(
    source=ontology["source"], target=ontology["target"]
)


print("========encoded_inputs========", encoded_inputs)
print("========source========", ontology["source"][0])
print("================", config.BERTRetrieval)

# init BERTRetrieval
model = BERTRetrieval(
    top_k=config.BERTRetrieval["top_k"], device=config.BERTRetrieval["device"]
)
# generate results
predicts = model.generate(input_data=encoded_inputs)

# post-processing
predicts = process.eval_preprocess_ir_outputs(predicts=predicts)

# evaluation
results = evaluator(
    track="anatomy", predicts=predicts, references=ontology["reference"]
)


# change
