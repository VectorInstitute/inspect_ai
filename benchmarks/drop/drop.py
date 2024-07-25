"""DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs

Dheeru Dua, Yizhong Wang, Pradeep Dasigi,
Gabriel Stanovsky, Sameer Singh, and Matt Gardner
https://arxiv.org/pdf/1903.00161

Based on: https://github.com/openai/simple-evals/blob/main/drop_eval.py
and https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/drop

# run with default fewshots (3)
inspect eval drop/drop.py

# run with more or no fewshots
inspect eval drop/drop.py -T fewshot=5
inspect eval drop/drop.py -T fewshot=false

# run without a prompt template
inspect eval drop/drop.py -T no_prompt_template=true
"""

import re
import string
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ModelName
from inspect_ai.scorer import Score, Target, bootstrap_std, mean, scorer
from inspect_ai.solver import (
    Generate,
    TaskState,
    generate,
    prompt_template,
    solver,
    system_message,
)


# Copied from
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/drop/utils.py#L23C1-L49C19
def get_answers(doc: Dict) -> List:
    def _flatten_validated_answers(validated_answers: Dict) -> List[Dict]:
        """Flattens a dict of lists of validated answers.

        {"number": ['1', '8'], ...}
        -> [{"number": ['1'], ...}, {"number": ['8'], ...}]
        """
        valid_answers = []
        for i in range(len(validated_answers["number"])):
            valid_answers.append(
                {
                    "number": validated_answers["number"][i],
                    "date": validated_answers["date"][i],
                    "spans": validated_answers["spans"][i],
                }
            )
        return valid_answers

    answers = []
    answers_set = set()
    candidates = [doc["answer"]] + _flatten_validated_answers(doc["validated_answers"])
    for candidate in candidates:
        answer = parse_answer(candidate)
        if answer in answers_set:
            continue
        answers_set.add(answer)
        answers.append(answer)
    return answers


def parse_answer(answer: Dict) -> Tuple:
    # NOTE: Everything is returned as a tuple for uniformity and hashability.
    if answer["number"] != "":
        return (str(answer["number"]),)
    if answer["spans"] != []:
        return tuple(answer["spans"])
    return (
        " ".join(
            [answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]
        ).strip(),
    )


# From here till '_normalize()' is copied from
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/drop/utils.py,
# and all functions are converted to async functions.
async def process_results(preds: Union[str, List[str]], golds: List) -> Tuple[float]:
    max_em = 0
    max_f1 = 0
    for gold_answer in golds:
        exact_match, f1_score = await get_metrics(preds, gold_answer)
        if gold_answer[0].strip():
            max_em = max(max_em, exact_match)
            max_f1 = max(max_f1, f1_score)
    return (max_em, max_f1)


async def get_metrics(predicted: Union[str, List[str]], gold: Union[str, List[str], Tuple[str]]) -> Tuple[float]:
    """Takes a predicted answer and a gold answer (that are both either a string or a list of strings), and returns exact match and the DROP F1 metric for the prediction.

    If you are writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = await _answer_to_bags(predicted)
    gold_bags = await _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(
        gold_bags[0]
    ):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = await _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


async def _answer_to_bags(answer: Union[str, List[str], Tuple[str]]) -> Tuple[List]:
    if isinstance(answer, list | tuple):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = await _normalize(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


async def _align_bags(predicted: List[set], gold: List[set]) -> np.ndarray:
    """Takes gold and predicted answer sets and first finds the optimal 1-1 alignment between them and gets maximum metric values over all the answers."""
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            match_numbers = await _match_numbers_if_present(gold_item, pred_item)
            if match_numbers:
                scores[gold_index, pred_index] = await _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


async def _compute_f1(predicted_bag: set, gold_bag: set) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


async def _match_numbers_if_present(gold_bag: set, predicted_bag: set) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        is_number = await _is_number(word)
        if is_number:
            gold_numbers.add(word)
    for word in predicted_bag:
        is_number = await _is_number(word)
        if is_number:
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


async def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


async def _remove_articles(text: str) -> str:
    _ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return _ARTICLES.sub(" ", text)


async def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


async def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    is_number = await _is_number(text)
    if not is_number:
        return "".join(ch for ch in text if ch not in exclude)
    else:
        return text


async def _fix_number(text: str) -> str:
    is_number = await _is_number(text)
    if is_number:
        return str(float(text))
    else:
        return text


async def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


async def _normalize(answer: str) -> str:
    tokens = []
    tokenized_answer = await _tokenize(answer)
    for token in tokenized_answer:
        token = await _remove_punc(token.lower())
        token = await _fix_number(token)
        token = await _remove_articles(token)
        token = await _white_space_fix(token)
        tokens.append(token)
    tokens = [token for token in tokens if token.strip()]
    normalized = " ".join(tokens).strip()
    return normalized


def format_input(doc: Dict) -> str:
    passage_str = f'''Passage: {doc["passage"]}'''
    question_str = f'''Question: {doc["question"]}'''
    answer_str = '''Answer:'''
    input_str = "\n".join([passage_str, question_str, answer_str])
    return input_str


def format_target(doc: Dict) -> List[str]:
    target = get_answers(doc)
    # Convert each tuple to str, since 'target' only accepts 'str' or 'List[str]'.
    target = ["|".join(elm) for elm in target]
    return target


def record_to_sample(record: Dict) -> Sample:
    return Sample(
        input=format_input(record),
        target=format_target(record),
        id=record["query_id"],
    )


def sample_to_fewshot(sample: Sample) -> str:
    target = sample.target[0].split("|")[0]
    return (f"""{sample.input} {target}""")


@scorer(metrics=[mean(), bootstrap_std()])
def drop_f1_scorer():

    async def score(state: TaskState, target: Target) -> Score:

        # Get generated answer
        answer = state.output.completion

        # Get target answers
        ref_answers = target.target
        # Convert str elm to tuple by splitting on "|"
        ref_answers = [tuple(elm.split("|")) for elm in ref_answers]

        em_score, f1_score = await process_results(answer, ref_answers)

        return Score(
            value=f1_score,
            answer=answer,
            metadata={
                "metrics": {
                    "f1_score": f1_score,
                    "em_score": em_score,
                },
                "gold_answers": ref_answers,
            }
        )

    return score


@solver
def drop_output_parser(model: ModelName, no_prompt_template: bool):

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        answer = state.output.completion
        state.metadata = {
            "raw_completion": answer,
        }

        match = re.search(ANSWER_PATTERN, answer)
        extracted_answer = match.group(1) if match else answer

        if no_prompt_template:
            extracted_answer = extracted_answer.split("\n\n")[0].strip()
        else:
            match model:
                case "Meta-Llama-3-8B-Instruct" | "Meta-Llama-3-70B-Instruct":
                    extracted_answer = extracted_answer.split("<|eot_id|>")[0]

        state.output.completion = extracted_answer

        return state

    return solve


def wrap_user_msg(prompt: str, model: ModelName) -> str:
    match model:
        case "Meta-Llama-3-8B-Instruct" | "Meta-Llama-3-70B-Instruct":
            prompt = """<|start_header_id|>user<|end_header_id|>\n\n""" + \
                prompt.strip("\n") + \
                """<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"""
    return prompt


def wrap_sys_msg(prompt: str, model: ModelName) -> str:
    match model:
        case "Meta-Llama-3-8B-Instruct" | "Meta-Llama-3-70B-Instruct":
            prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n""" + \
                prompt.strip("\n") + """<|eot_id|>"""
    return prompt


# Borrowed from: https://github.com/openai/simple-evals/blob/main/common.py#L24
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


@task
def drop(
    model: ModelName,
    no_prompt_template: bool=False,
    fewshot: int=3,
    fewshot_seed: int=42
) -> Task:
    """Inspect task implementing the DROP benchmark.

    Arguments:
        model (ModelName): Name (or path) of the model being used.
        no_prompt_template (bool): Set to true for not applying any
            prompt template. Default is false.
        fewshot (int): Number of few shot examples to use.
        fewshot_seed (int): Random seed for sampling few shot examples.
    """
    if not no_prompt_template:
        # Get prompt
        prompt_dict = DROP_PROMPT_TEMPLATE

        # Apply model specific templates
        prompt_user = wrap_user_msg(prompt_dict["user"], model)
        prompt_sys = wrap_sys_msg(prompt_dict["system"], model)
        prompt_sys_w_examples = wrap_sys_msg(prompt_dict["system_w_examples"], model)
    else:
        prompt_user = """{prompt}"""
        prompt_sys = ""
        prompt_sys_w_examples = """{examples}\n\n"""

    plan = [prompt_template(prompt_user), generate(), drop_output_parser(model, no_prompt_template)]

    if fewshot:
        fewshots = hf_dataset(
            "EleutherAI/drop",
            split="train",
            trust=True,
            sample_fields=record_to_sample,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        plan.insert(
            0,
            system_message(
                prompt_sys_w_examples.format(
                    examples="\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
                )
            )
        )
    else:
        if prompt_sys != "":
            plan.insert(
                0,
                system_message(prompt_sys)
            )

    return Task(
        dataset=hf_dataset(
            "EleutherAI/drop",
            split="validation",
            trust=True,
            sample_fields=record_to_sample,
        ),
        plan=plan,
        scorer=drop_f1_scorer(),
    )


DROP_PROMPT_TEMPLATE = {
    # Based on the prompt provided here: https://github.com/openai/simple-evals/blob/main/drop_eval.py#L261C13-L283C91
    "system": """You will be asked to read a passage and answer a question.\n\n""",
    "system_w_examples": """You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.\n\n""" + \
        """Examples\n{examples}\n\n""",
    "user": """Your Task\n---\n{prompt}\n\n""" + \
        """Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.\n""",
}
