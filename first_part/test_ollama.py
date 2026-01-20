# ==========================================

# 想读取一部分的xml file文件, 并用ollama的mistral模型进行处理, 输出json格式的数据
# {'name': 'MA_0000001', 'iri': 'http://mouse.owl#MA_0000001', 'label': 'mouse anatomy', 'childrens': [], 'parents': [], 'synonyms': [], 'comment': []}
# 然后计算F1 score

# 下一步: 下一步就是使用prompt engineer写一个code来提取ontology class的信息,并且输出json格式的数据

# =========================================


# from ollama import chat
# from ollama import ChatResponse
# import time
# import ollama
# import json
# from pathlib import Path
# start = time.time()

import json
import ollama
from typing import Dict, Any


import ollama
import json


def extract_ontology_class(ontology_snippet, model="mistral:7b"):
    system_prompt = f"""
You are an ontology engineer.

You will receive a snippet of an OWL ontology (in XML or Turtle).
Your task is to extract information about ONE ontology class and
return it as a JSON object with a fixed schema.

Rules:
- Focus on a single main class in the snippet.
- If some fields are missing, use empty list [] or null.
- Output ONLY valid JSON.
- No markdown.
- No comments.
- No trailing text.
- The output MUST be parseable by Python json.loads().
"""

    user_prompt = f"""
Below is an OWL ontology snippet:

{ontology_snippet}

Output a single JSON object with the following keys:
{{
  "name": "string (the OWL class ID, e.g. MA_0000001)",
  "iri": "string (the OWL class IRI)",
  "label": "string (rdfs:label of the class)",
  "childrens": ["list of strings (IRIs or IDs of direct subclasses)"],
  "parents": ["list of strings (IRIs or IDs of direct superclasses)"],
  "synonyms": ["list of strings (synonym labels, if any)"],
  "comment": ["list of strings (comments/definitions, if any)"]
}}

Return ONLY valid JSON.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response["message"]["content"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("Model did not return valid JSON:\n" + raw)

    return data


owl_snippet = """
<Class rdf:about="http://mouse.owl#MA_0000001">
    <rdfs:label>mouse anatomy</rdfs:label>
</Class>
"""

result = extract_ontology_class(owl_snippet)
print(result)
