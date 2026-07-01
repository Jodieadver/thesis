from ontomap.ontology import OMIMORDODiseaseOMDataset
from ontomap.base import BaseConfig
from ontomap.evaluation.evaluator import evaluator
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval
from ontomap.postprocess import process
import subprocess
import xml.etree.ElementTree as ET
from typing import List

#  original code
# Setting configurations for experimenting 'retrieval' on CPU
config = BaseConfig(approach="retrieval").get_args(device="cpu")
# set dataset directory
config.root_dir = "datasets"
# parse task source, target, and reference ontology
ontology = OMIMORDODiseaseOMDataset().collect(root_dir=config.root_dir)

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
print("========predicts========")
for predict in predicts:
    print(predict)

predicts = list(filter(lambda x: x["score"] > 0.9, predicts))

# evaluation
results = evaluator(
    track="bio-ml", predicts=predicts, references=ontology["reference"]
)
print("========results========", results)

# ################# add function: get a small testset of matching results

# _RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
# _RDFS = "http://www.w3.org/2000/01/rdf-schema#"
# _OWL = "http://www.w3.org/2002/07/owl#"


# def parse_owl(path: str) -> dict:
#     """Index an OWL/XML file by IRI → {label, subClassOf, restrictions}."""
#     root = ET.parse(path).getroot()
#     index = {}
#     for cls in root.findall(f".//{{{_OWL}}}Class"):
#         iri = cls.get(f"{{{_RDF}}}about")
#         if not iri:
#             continue
#         label_el = cls.find(f"{{{_RDFS}}}label")
#         sub_of, restrictions = [], []
#         for sc in cls.findall(f"{{{_RDFS}}}subClassOf"):
#             ref = sc.get(f"{{{_RDF}}}resource")
#             if ref:
#                 sub_of.append(ref)
#             restr = sc.find(f"{{{_OWL}}}Restriction")
#             if restr is not None:
#                 prop = restr.find(f"{{{_OWL}}}onProperty")
#                 val = restr.find(f"{{{_OWL}}}someValuesFrom")
#                 if prop is not None and val is not None:
#                     p = prop.get(f"{{{_RDF}}}resource")
#                     v = val.get(f"{{{_RDF}}}resource")
#                     if p and v:
#                         restrictions.append({"property": p, "target": v})
#         index[iri] = {
#             "label": label_el.text if label_el is not None else "",
#             "subClassOf": sub_of,
#             "restrictions": restrictions,
#         }
#     return index


# def extract_small_testset(
#     predicts: List, references: List, source_owl_path: str, target_owl_path: str
# ) -> List:
#     """
#     :param predicts:
#         [{
#             "source": ...,
#             "target": ...,
#             "score": ...
#         }, ...]
#     :param references:
#         [{
#             "source": ...,
#             "target": ...,
#             "relation": ...
#         }, ...]
#     :return: intersection
#     """
#     source_index = parse_owl(source_owl_path)
#     target_index = parse_owl(target_owl_path)

#     testset = []
#     for predict in predicts:
#         if len(testset) >= 5:
#             break
#         for reference in references:
#             if (
#                 predict["source"] == reference["source"]
#                 and predict["target"] == reference["target"]
#             ):
#                 src_iri = predict["source"]
#                 tgt_iri = predict["target"]
#                 testset.append(
#                     {
#                         "source": src_iri,
#                         "target": tgt_iri,
#                         "score": predict["score"],
#                         "relation": reference["relation"],
#                         "source_owl": source_index.get(src_iri, {}),
#                         "target_owl": target_index.get(tgt_iri, {}),
#                     }
#                 )
#                 break
#     return testset


# print("========results========", results)

# testset = extract_small_testset(
#     predicts=predicts,
#     references=ontology["reference"],
#     source_owl_path="datasets/anatomy/mouse-human-test/source.xml",
#     target_owl_path="datasets/anatomy/mouse-human-test/target.xml",
# )
# print("========testset========", testset)

# print(
#     "========testset source ========",
#     [item["source"] for item in testset],
# )

# print(
#     "========testset target ========",
#     [item["target"] for item in testset],
# )
