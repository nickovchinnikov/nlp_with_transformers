# %% [markdown]
# ### First look at Hugging Face Datasets
# %%
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from datasets import (
    list_datasets,
    load_dataset
)
from transformers import (
    AutoTokenizer,
    DistilBertTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from huggingface_hub import login

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

# %%
all_datasets = list_datasets()
print(f"There are {len(all_datasets)} datasets currently available on the Hub")
print(f"The first 10 are: {all_datasets[:10]}")

# %%
emotions = load_dataset("emotion")
emotions

# %%
train_ds = emotions["train"]
train_ds

# %%
# We can use an access to the collection as usual list
len(train_ds)

# %%
train_ds[0]

# %%
train_ds.column_names

# %%
train_ds.features

# %%
train_ds[:5]

# %% [markdown]
# ### What If My Dataset Is Not on the Hub?
# %%
# Download data with wget
dataset_url = "https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt"

os.system(f"""
    wget {dataset_url}
""")

# %%
# Load dataset locally
# emotions_local = load_dataset(
#     "csv",
#     data_files="train.txt",
#     sep=";",
#     names=["text", "label"]
# )

# %%
# Load from remote source
# dataset_url = "https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt?dl=1"

# emotions_remote = load_dataset(
#     "csv",
#     data_files=dataset_url,
#     sep=";",
#     names=["text", "label"]
# )

# %% [markdown]
# ### From datasets to DataFrames
# %%
emotions.set_format(type="pandas")
df = emotions["train"][:]

df.head()

# %%
# Change label from int to string
def label_int2str(int_label):
    return emotions["train"].features["label"].int2str(int_label)

# %%
df["label_name"] = df["label"].apply(label_int2str)
df.head()

# %%
df["label_name"] = df["label"].apply(
    emotions["train"].features["label"].int2str
)
df.head()

# %% [markdown]
# ### Looking at the Class Distribution
# %%
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# %% [markdown]
# ### How long are our tweets?
# %%
df["words_per_text"] = df["text"].str.split().apply(len)
df.boxplot(
    "words_per_text",
    by="label_name",
    showfliers=False,
    grid=False,
    color="black"
)
plt.suptitle("")
plt.xlabel("")
plt.show()

# %%
# reset output format for the dataset
emotions.reset_format()

# %% [markdown]
# ### From text to tokens
# #### Char tokens
# %%
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
tokenized_text

# %%
token2idx = dict(
    zip(
        sorted(set(tokenized_text)), itertools.count()
    )
)
token2idx

# %%
# Take input ids
input_ids = list(
    map(token2idx.get, tokenized_text)
)
input_ids

# %% [markdown]
# One-hot vectors
# %%
categorical_df = pd.DataFrame({
    "Name": ["Bumblebee", "Optimus Prime", "Megatron"],
    "Label ID": [0, 1, 2]
})
categorical_df

# %%
# Pandas get_dummies creates ont-hot
pd.get_dummies(categorical_df["Name"])

# %%
# One-hot encoding with pytorch
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
one_hot_encodings.shape

# %%
print(f"""Token: {tokenized_text[0]}
Tensor index: {input_ids[0]}
One-hot: {one_hot_encodings[0]}
""")

# %% [markdown]
# ### Word tokenization
# %%
tokenized_text = text.split()
tokenized_text

# %% [markdown]
### Subword Tokenization
# %%
# automatically retrieve the model’s configuration,
# pretrained weights, or vocabulary from 
# the name of the checkpoint
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# %%
# Or select a specific tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

# %%
encoded_text = tokenizer(text)
encoded_text

# %%
tokens = tokenizer.convert_ids_to_tokens(encoded_text["input_ids"])
tokens

# %%
tokenizer.convert_tokens_to_string(tokens)

# %%
tokenizer.vocab_size

# %%
tokenizer.model_max_length

# %%
tokenizer.model_input_names

# %% [markdown]
# ### Tokenize the whole dataset
# %%
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"], truncation=True, padding=True)

# %%
tokenize_dataset(emotions["train"][:2])

# %%
emotions_encoded = emotions.map(tokenize_dataset, batched=True)
emotions_encoded

# %%
emotions_encoded["train"].column_names

# %% [markdown]
# ### Training a text classifier
# %%
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load pretrained model
model = AutoModel.from_pretrained(model_ckpt).to(device)

# %%
# Extracting the last hidden state
text = "this is a text"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape {inputs['input_ids'].size()}")

# %%
# Overwrite the inputs
inputs = tokenizer(text, return_tensors="pt")
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)

outputs

# %%
# Overwrite the inputs
inputs = tokenizer(text, return_tensors="pt")
inputs = {k:v.to(device) for k,v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

outputs

# %%
outputs.last_hidden_state.shape

# %%
outputs.last_hidden_state[:,0].shape

# %%
def extract_hidden_state(batch):
    inputs = {
        k:v.to(device) for k,v in batch.items()
        if k in tokenizer.model_input_names
    }
              
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        
    # Return vector for [CLS] token
    return {
        "hidden_state": last_hidden_state[:,0].cpu().numpy()
    }

# %%
emotions_encoded.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

emotions_encoded

# %%
emotions_hidden = emotions_encoded.map(
    extract_hidden_state, batched=True
)
emotions_hidden

# %%
emotions_hidden["train"].column_names

# %% [markdown]
# #### Create a feature matrix
# %%
X_train, X_valid, y_train, y_valid = \
    np.array(emotions_hidden["train"]["hidden_state"]), \
    np.array(emotions_hidden["validation"]["hidden_state"]), \
    np.array(emotions_encoded["train"]["label"]), \
    np.array(emotions_encoded["validation"]["label"])

X_train.shape, X_valid.shape

# %% [markdown]
# #### Visualizing the training set
# %%
# Scale features to [0,1] range
X_train_scaled = MinMaxScaler().fit_transform(X_train)

# %%
# Initialize and fit UMAP
mapper = UMAP(
    n_components=2, metric="cosine"
).fit_transform(X_train_scaled)

# %%
# Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(
    mapper,
    columns=["X", "Y"]
)
df_emb["label"] = y_train
df_emb.head()

# %%
fig, ax = plt.subplots(2, 3, figsize=(7, 5))
axes = ax.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

for i, (cmap, label) in enumerate(zip(cmaps, labels)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(
        df_emb_sub["X"],
        df_emb_sub["Y"],
        cmap=cmap,
        gridsize=20
    )
    axes[i].set_title(label)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Train a simple classifier
# %%
lr_clf = LogisticRegression(
    max_iter=3000
)

lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)

# %%
dm_clf = DummyClassifier()

dm_clf.fit(X_train, y_train)
dm_clf.score(X_valid, y_valid)

# %%
def plot_confusion_matrix(y_pred, y_true, labels):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    cm_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    cm_disp.plot(
        cmap="Blues",
        values_format=".2f",
        ax=ax,
        colorbar=False
    )
    plt.title("Normalized Confusion Matrix")
    plt.show()

# %%
y_pred = lr_clf.predict(X_valid)

plot_confusion_matrix(
    y_pred,
    y_valid,
    labels
)


# %% [markdown]
# #### Loading a pretrained model
# %%
# Auth
login(token="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")

# %%
num_labels = 6
model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        model_ckpt,
        num_labels=num_labels
    )
    .to(device)
)

# %% [markdown]
# #### Defining the performance metric
# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1
    }

# %% [markdown]
# #### Training the model
# %%
batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=True,
    log_level="error"
)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer
)
trainer.train()

# %%
preds_output = trainer.predict(emotions_encoded["validation"])
preds_output.metrics

# %%
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)

# %% [markdown]
# ### Error analysis
# %%
# Fix model device, found a bug
num_labels = 6
model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        model_ckpt,
        num_labels=num_labels
    )
    .to(device)
)

emotions_encoded = emotions.map(tokenize_dataset, batched=True)

# %%
def forward_pass_with_label(batch):
    # Fix code from the book, didn't work without the fix `pad_sequence`
    inputs = {
        k: v.to(device)
        if torch.is_tensor(v)
        else pad_sequence(v, batch_first=True).to(device)
        for k,v in batch.items()
        if k in tokenizer.model_input_names
    }

    with torch.no_grad():
        outputs = model(**inputs)
        pred_labels = torch.argmax(outputs.logits, axis=-1)
        loss = F.cross_entropy(
            outputs.logits,
            batch["label"].to(device),
            reduction="none"
        )

        return {
            "loss": loss.cpu().numpy(),
            "predicted_label": pred_labels.cpu().numpy()
        }

# %%
# Create emotions_encoded tokenized again
# emotions_encoded = emotions.map(tokenize_dataset, batched=True)

# Convert dataset back to pytorch tensors
emotions_encoded.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

# %%
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label,
    batched=True,
    batch_size=16
)

# %%
# Create a DataFrame with the text, losses and predicted/true labels
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]

df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (
    df_test["predicted_label"]
    .apply(label_int2str)
)

# %%
df_test.sort_values("loss", ascending=False).head(10)

# %%
df_test.sort_values("loss", ascending=True).head(10)

# %% [markdown]
# #### Saving and sharing the model
# %%
trainer.push_to_hub(commit_message="Training completed!")

# %%
model_id = "nickovchinnikov/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

# %%
# Then let’s test the pipeline with a sample tweet:
custom_tweet = "I saw a movie today and it was really good."
preds = classifier(custom_tweet, return_all_scores=True)
preds

# %%
pred_df = pd.DataFrame(preds[0])
plt.bar(labels, pred_df["score"], color="C0")
plt.title(f"{custom_tweet}")
plt.ylabel("Class probability (%)")
plt.show()

# %%
