import csv
import logging

import pandas as pd
import pyconcepticon
import wandb
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


SWEEP = True
CONCEPTICON_PATH = "concepticon-data/"

wandb.init(project="concepticon")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average="micro")


def prepare_data_all_glosses():
    concepticon = pyconcepticon.Concepticon(CONCEPTICON_PATH)

    data = []
    labels = set()

    for conceptlist in concepticon.conceptlists.values():
        for concepts in conceptlist.concepts.values():
            if concepts.gloss and concepts.concepticon_id:
                data.append([concepts.gloss, int(concepts.concepticon_id)])
                labels.add(int(concepts.concepticon_id))
            if concepts.english and concepts.concepticon_id:
                data.append([concepts.english, int(concepts.concepticon_id)])
                labels.add(int(concepts.concepticon_id))

    return data, len(labels), {x: a for a, x in enumerate(labels)}


def prepare_data_mappings_only():
    data = []
    labels = set()

    with open("map-en.tsv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for i, row in enumerate(reader):
            labels.add(int(row["ID"]))
            data.append([row["GLOSS"].split("///")[1], int(row["ID"])])

    return data, len(labels), {x: a for a, x in enumerate(labels)}


# Helper to map labels used for training back to Concepticon_IDs
# e.g.: find_mapping(mappings, int(model.predict(["earlobe"])[0][0]))
def find_mapping(global_mappings, label):
    return [k for k, v in global_mappings.items() if v == label]


"""
Either use prepare_data_all_glosses() for training/evaluation with all glosses (i.e.
supersampling) or use prepare_data_mappings_only() for training/evaluation using only
GLOSS-CONCEPTICON_ID mappings.

mappings is a dictionary mapping 0 to n on Concepticon_IDs that can be used to resolve
model.predict() results.
"""
training_eval, num_of_labels, mappings = prepare_data_all_glosses()

df = training_eval
df = pd.DataFrame(df)
df.columns = ["text", "labels"]
df = df.replace({"labels": mappings})

train_df, eval_df = train_test_split(df, test_size=0.1)

sweep_config = {
    "name": "concepticon-transformers-sweep",
    "method": "bayes",
    "metric": {"name": "mcc", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0, "max": 4e-6},
        "num_train_epochs": {"min": 1, "max": 4},
    },
    "early_terminate": {"type": "hyperband", "min-iter": 6},
}

sweep_id = wandb.sweep(sweep_config, project="concepticon")

model_args = ClassificationArgs(
    num_train_epochs=6,
    learning_rate=4e-6,
    no_cache=True if SWEEP else False,
    no_save=True if SWEEP else False,
    save_eval_checkpoints=False if SWEEP else True,
    save_model_every_epoch=False if SWEEP else True,
    overwrite_output_dir=True,
    reprocess_input_data=True,
    evaluate_during_training=True,
    evaluate_during_training_silent=False,
    evaluate_during_training_steps=1000,
    wandb_project="concepticon",
    train_batch_size=15,
    eval_batch_size=10,
    use_early_stopping=True,
    early_stopping_delta=0.01,
    early_stopping_metric="mcc",
    early_stopping_metric_minimize=False,
    early_stopping_patience=5,
)


def train():
    wandb.init()
    model_args.wandb_kwargs = {"id": wandb.run.id}

    model = ClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=num_of_labels,
        use_cuda=True,
        args=model_args,
        sweep_config=wandb.config,
    )

    model.train_model(train_df, eval_df=eval_df, f1=f1_multiclass)

    result, model_outputs, wrong_predictions = model.eval_model(
        eval_df, f1=f1_multiclass, acc=accuracy_score
    )

    print(result, model_outputs)

    wandb.join()


wandb.agent(sweep_id, train)
