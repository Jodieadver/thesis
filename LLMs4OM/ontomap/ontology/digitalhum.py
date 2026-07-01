# -*- coding: utf-8 -*-
import os.path
from typing import Any, Dict, List

import rdflib
from rdflib.namespace import RDF, SKOS

from ontomap.base import BaseAlignmentsParser, BaseOntologyParser, OMDataset

track = "digitalhum"


def _local_name(iri: str) -> str:
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


class SKOSOntologyParser(BaseOntologyParser):
    """Parses SKOS thesauri (skos:Concept individuals), as opposed to
    BaseOntologyParser's owl:Class-based parsing used elsewhere."""

    def get_comments(self, owl_class: Any) -> List:
        return []

    def _label(self, graph: Any, concept: Any, prop: Any) -> str:
        labels = list(graph.objects(concept, prop))
        if not labels:
            return ""
        for label in labels:
            if getattr(label, "language", None) == "en":
                return str(label)
        return str(labels[0])

    def _literal_list(self, graph: Any, concept: Any, prop: Any) -> List[str]:
        return [str(o) for o in graph.objects(concept, prop)]

    def _related_concepts(self, graph: Any, concept: Any, prop: Any) -> List[Dict]:
        related = []
        for obj in graph.objects(concept, prop):
            related.append(
                {
                    "iri": str(obj),
                    "name": _local_name(str(obj)),
                    "label": self._label(graph, obj, SKOS.prefLabel) or str(obj),
                }
            )
        return related

    def parse(self, root_dir: str, ontology_file_name: str) -> List:
        input_file_path = os.path.join(root_dir, ontology_file_name)
        print(f"\t\tworking on {input_file_path}")
        graph = rdflib.Graph()
        graph.parse(input_file_path)

        parsed_concepts = []
        for concept in set(graph.subjects(RDF.type, SKOS.Concept)):
            label = self._label(graph, concept, SKOS.prefLabel)
            if not label:
                continue
            comments = self._literal_list(graph, concept, SKOS.definition)
            comments += self._literal_list(graph, concept, SKOS.scopeNote)
            parsed_concepts.append(
                {
                    "name": _local_name(str(concept)),
                    "iri": str(concept),
                    "label": label,
                    "childrens": self._related_concepts(graph, concept, SKOS.narrower),
                    "parents": self._related_concepts(graph, concept, SKOS.broader),
                    "synonyms": self._literal_list(graph, concept, SKOS.altLabel),
                    "comment": comments,
                }
            )
        return parsed_concepts


class DigitalHumOMDataset(OMDataset):
    track = track
    source_ontology = SKOSOntologyParser()
    target_ontology = SKOSOntologyParser()
    alignments: BaseAlignmentsParser = BaseAlignmentsParser()

    def collect(self, root_dir: str) -> Dict:
        om_root_path = os.path.join(root_dir, self.track, self.ontology_name)
        data = {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source": self.source_ontology.parse(
                root_dir=om_root_path, ontology_file_name="source.rdf"
            ),
            "target": self.target_ontology.parse(
                root_dir=om_root_path, ontology_file_name="target.rdf"
            ),
            "reference": self.alignments.parse(
                root_dir=om_root_path, reference_file_name="reference.rdf"
            ),
        }
        return data


class DefcPactolsOMDataset(DigitalHumOMDataset):
    ontology_name = "arch1_defc-pactols"
    working_dir = os.path.join(track, ontology_name)


class IdaiPactolsOMDataset(DigitalHumOMDataset):
    ontology_name = "arch2_idai-pactols"
    working_dir = os.path.join(track, ontology_name)


class IronageDanubePactolsOMDataset(DigitalHumOMDataset):
    ontology_name = "arch3_ironagedanube-pactols"
    working_dir = os.path.join(track, ontology_name)


class PactolsParthenosOMDataset(DigitalHumOMDataset):
    ontology_name = "arch4_pactols-parthenos"
    working_dir = os.path.join(track, ontology_name)


class IdaiParthenosOMDataset(DigitalHumOMDataset):
    ontology_name = "cult1_idai-parthenos"
    working_dir = os.path.join(track, ontology_name)


class OeaiParthenosOMDataset(DigitalHumOMDataset):
    ontology_name = "cult2_oeai-parthenos"
    working_dir = os.path.join(track, ontology_name)
