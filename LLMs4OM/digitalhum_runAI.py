import csv
import os
import time

from openai import OpenAI

from ontomap.ontology import ontology_matching
from ontomap.base import BaseConfig
from ontomap.evaluation.evaluator import evaluator
from ontomap.encoder import IRILabelInRAGEncoder
from ontomap.ontology_matchers.rag.rag import RAG, RAGBasedOpenAILLMArch
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval
from ontomap.postprocess import process

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
RESULTS_PATH = "my_test_result/digitalhum_AI_results.csv"
MODEL_NAME = "gpt-oss-120b on cerebras"


class CerebrasGPTOSS120BLLM(RAGBasedOpenAILLMArch):
    """Chat-completion LLM arch backed by Cerebras' OpenAI-compatible API,
    serving openai/gpt-oss-120b. Reuses RAGBasedOpenAILLMArch's prompt
    handling and yes/no post-processing — only the client/model differ from
    ChatGPTOpenAILLM.
    """

    path = "gpt-oss-120b"

    def __init__(self, **kwargs) -> None:
        # Bypass LLM.__init__: it unconditionally requires OPENAI_KEY plus
        # local itemset.txt/explanations.txt files used by an unrelated
        # explanation-augmented prompt path. This arch only needs an
        # OpenAI-compatible client pointed at Cerebras.
        self.kwargs = kwargs
        self.dictionary = {}
        self.client = OpenAI(
            api_key=os.environ["CEREBRAS_API_KEY"],
            base_url=CEREBRAS_BASE_URL,
        )

    def __str__(self):
        return "CerebrasGPTOSS120BLLM"

    def generate_for_one_input(self, tokenized_input_data):
        # gpt-oss-120b is a reasoning model: pass reasoning_effort="low" so it
        # spends fewer tokens thinking and leaves room for the actual answer
        # within our small max_tokens budget.
        if len(tokenized_input_data[0].split(", ")) > 1000:
            tokenized_input_data[0] = ", ".join(
                tokenized_input_data[0].split(", ")[:1000]
            )
        prompt = [{"role": "user", "content": tokenized_input_data[0]}]
        is_generated_output = False
        response = None
        while not is_generated_output:
            try:
                response = self.client.chat.completions.create(
                    model=self.path,
                    messages=prompt,
                    temperature=self.kwargs["temperature"],
                    max_tokens=self.kwargs["max_token_length"],
                    reasoning_effort="low",
                )
                is_generated_output = True
            except Exception as error:
                print(
                    f"Unexpected {error}, {type(error)} \n"
                    f"Going for sleep for {self.kwargs['sleep']} seconds!"
                )
                time.sleep(self.kwargs["sleep"])
        return [response]

    def post_processor(self, generated_texts):
        sequences, sequence_probas = [], []
        for generated_text in generated_texts:
            content = generated_text.choices[0].message.content
            processed_output = (content or "").lower()
            proba = 1
            sequences.append("yes" if "yes" in processed_output else "no")
            sequence_probas.append(proba)
        return [sequences, sequence_probas]


class CerebrasGPTOSSBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = CerebrasGPTOSS120BLLM

    def __str__(self):
        return super().__str__() + "-CerebrasGPTOSSBertRAG"


# Setting configurations for experimenting 'rag' on CPU with batch size of 16
config = BaseConfig(approach="rag").get_args(device="cpu", batch_size=16)
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
    encoded_inputs = IRILabelInRAGEncoder()(
        source=ontology["source"], target=ontology["target"]
    )

    # init Cerebras gpt-oss-120b + BERT, replacing MistralLLMBertRAG
    model = CerebrasGPTOSSBertRAG(
        retriever_config=config.MistralBertRAG["retriever-config"],
        llm_config={
            "max_token_length": 100,
            "temperature": 0,
            "top_p": 0.95,
            "sleep": 5,
            "batch_size": 16,
        },
    )

    # generate results
    predicts = model.generate(input_data=encoded_inputs)

    # post-processing
    predicts, _ = process.postprocess_hybrid(
        predicts=predicts, llm_confidence_th=0.7, ir_score_threshold=0.9
    )
    # evaluation
    results = evaluator(
        track="digitalhum", predicts=predicts, references=ontology["reference"]
    )
    elapsed = round(time.time() - start_time, 2)
    print(f"========Time cost for {dataset.ontology_name}========", elapsed)
    print(f"{dataset.ontology_name}: {results}")

    results_writer.writerow([dataset.ontology_name, MODEL_NAME, elapsed, results])
    results_file.flush()

results_file.close()
print(f"\nResults written to {RESULTS_PATH}")
