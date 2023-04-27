# %%
import pandas as pd
from transformers import pipeline

# %% [markdown]
# ### A tour of Transformers applications
# %%
text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# %%
classifier = pipeline("text-classification")

# %%
outputs = classifier(text)
pd.DataFrame(outputs)

# %% [markdown]
# ### Named entity recognition (NER)
# %%
ner_tag = pipeline(task="ner", aggregation_strategy="simple")

# %%
outputs = ner_tag(text)
pd.DataFrame(outputs)

# %% [markdown]
# ### Question answering (QA)
# %%
reader = pipeline("question-answering")

# %%
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])

# %% [markdown]
# ### Summarization
# %%
summarizer = pipeline("summarization")

# %%
outputs = summarizer(text, clean_up_tokenization_spaces=True)
outputs[0]['summary_text']

# %% [markdown]
# ### Translation
# %%
translator = pipeline(
    "translation_en_to_de",
    model="Helsinki-NLP/opus-mt-en-de"
)

# %%
outputs = translator(
    text, clean_up_tokenization_spaces=True, max_length=100
)
outputs[0]['translation_text']

# %% [markdown]
# ### Text-Generation
# %%
generator = pipeline("text-generation")

# %%
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
outputs[0]['generated_text']

# %%
