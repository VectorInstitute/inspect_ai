import os
import pandas as pd

from datasets import load_dataset

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import model_graded_bias
from inspect_ai.solver import generate


def load_and_save_bold_dataset(
    save_folder: str = "datasets"
) -> None:
    # load dataset from huggingface
    bold_dataset = load_dataset("AlexaAI/bold", split="train")

    # expand and convert to pandas dataframe
    bold_df = []
    for record in bold_dataset:
        domain = record["domain"]
        name = record["name"]
        category = record["category"]
        uid_prefix = f"{domain}_{category}_{name}"

        prompts = record["prompts"]
        sources = record["wikipedia"]
        for prompt_idx, prompt in enumerate(prompts):
            bold_df.append(
                {
                    "uid": f"{uid_prefix}_{prompt_idx}",
                    "domain": domain,
                    "name": name,
                    "category": category,
                    "prompt": prompt,
                    "wikipedia": sources[prompt_idx],
                }
            )
    bold_df = pd.DataFrame.from_records(bold_df)
    assert bold_df["uid"].nunique() == len(bold_df)   

    # save to csv
    bold_df.to_csv(f"{save_folder}/bold.csv", index=False)


# map records to inspect sample
def record_to_sample(record):
    return Sample(
        id=record["uid"],
        input=record["prompt"],
        metadata={
            "domain": record["domain"],
            "name": record["name"],
            "category": record["category"],
            "wikipedia": record["wikipedia"],
        }
    )


@task
def bold_bias(
    limit: int = None,
    seed: int = 37,
):
    if not os.path.exists("datasets/bold.csv"):
        load_and_save_bold_dataset()
    dataset = csv_dataset(
        csv_file="datasets/bold.csv",
        sample_fields=record_to_sample,
        shuffle=True,
    )
    if limit is not None:
        dataset = dataset[limit]

    grader_model = get_model(
        "openai/gpt-4",
        config=GenerateConfig(
            temperature=0,
            max_connections=10,
            seed=seed,
        )
    )

    return Task(
        dataset=dataset,
        plan=[generate()],
        scorer=model_graded_bias(
            model=grader_model,
        ),
    )
