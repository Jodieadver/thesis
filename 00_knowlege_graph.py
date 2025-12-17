# =======================
# 1. creates a knowledge graph based on words in multiple different knowledge bases (text documents).
# =======================

from neo4j import GraphDatabase
import spacy

# ----------------------------
# Neo4j connection
# ----------------------------
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
DATABASE = "demo"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ----------------------------
# NLP model
# ----------------------------
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# Example documents
# ----------------------------
documents = {
    "doc1": """
    Python is a programming language.
    Python is widely used in data science and machine learning.
    """,
    "doc2": """
    Neo4j is a graph database.
    Graph databases store data as nodes and relationships.
    Neo4j is often used with Python.
    """,
    "doc3": """
    Machine learning uses algorithms and data.
    Data science often relies on Python and graph databases.
    """,
}


# ----------------------------
# Helper functions
# ----------------------------
def create_document(tx, doc_id, text):
    tx.run(
        """
        MERGE (d:Document {id: $id})
        SET d.text = $text
        """,
        id=doc_id,
        text=text,
    )


def create_entity(tx, name):
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        """,
        name=name,
    )


def link_doc_entity(tx, doc_id, entity_name):
    tx.run(
        """
        MATCH (d:Document {id: $doc_id})
        MATCH (e:Entity {name: $entity})
        MERGE (d)-[:MENTIONS]->(e)
        """,
        doc_id=doc_id,
        entity=entity_name,
    )


def relate_entities(tx, e1, e2):
    if e1 == e2:
        return
    tx.run(
        """
        MATCH (a:Entity {name: $e1})
        MATCH (b:Entity {name: $e2})
        MERGE (a)-[:RELATED_TO]->(b)
        """,
        e1=e1,
        e2=e2,
    )


def create_word(tx, word):
    tx.run(
        """
        MERGE (w:Word {text: $word})
        """,
        word=word,
    )


def link_entity_word(tx, entity, word):
    tx.run(
        """
        MATCH (e:Entity {name: $entity})
        MATCH (w:Word {text: $word})
        MERGE (e)-[:HAS_WORD]->(w)
        """,
        entity=entity,
        word=word,
    )


# ----------------------------
# Graph construction
# ----------------------------
with driver.session(database=DATABASE) as session:
    for doc_id, text in documents.items():
        session.execute_write(create_document, doc_id, text)

        doc = nlp(text)

        for sent in doc.sents:
            # Extract entities: named entities + noun chunks
            entities = set()

            for ent in sent.ents:
                entities.add(ent.text)

            for chunk in sent.noun_chunks:
                if chunk.root.pos_ in {"NOUN", "PROPN"}:
                    entities.add(chunk.text)

            # Create entity nodes
            for entity in entities:
                session.execute_write(create_entity, entity)
                session.execute_write(link_doc_entity, doc_id, entity)

                # Words inside entity
                for token in nlp(entity):
                    if token.is_alpha:
                        session.execute_write(create_word, token.text.lower())
                        session.execute_write(
                            link_entity_word, entity, token.text.lower()
                        )

            # Relate co-occurring entities in the same sentence
            for e1 in entities:
                for e2 in entities:
                    session.execute_write(relate_entities, e1, e2)

driver.close()
