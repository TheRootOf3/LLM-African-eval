from __future__ import annotations
import os


import glob
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
import torch
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
    # else "cpu"
)

print(f"Using device: {device}")

LIMITED_LANGS: list[str] = ["amh", "hau", "ibo", "swa", "yor"]


def load_model(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir="../.cached_models"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../.cached_models")

    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.to(device)

    return model, tokenizer


# script from Nikita Vassilyev and Alex Pejovic
def prompt_llm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    message: str,
    temperature: float = 0.7,
    repetition_penalty: float = 1.176,
    top_p: float = 0.1,
    top_k: int = 40,
    num_beams: int = 1,
    max_new_tokens: int = 512,
):
    inputs = tokenizer(message, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[-1] > model.config.n_positions:
        print(f"Input too long, exceeds {model.config.n_positions} tokens")
        return None

    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

    generation_config = GenerationConfig(
        ### temperature, top_p, and top_k are not needed since we are using 1 beam
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    # result = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output[len(message) :]


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def getlabel_string(filename):
    with open(filename) as f:
        label_list = f.read().splitlines()
    label_string = label_list[0]
    for i, value in enumerate(label_list[:-2], 1):
        label_string += ", " + label_list[i]

    label_string += " or " + label_list[-1]

    return label_string, label_list


def get_language(files, senti=False, mt=False):

    if senti:
        lang = sorted([i.split("/")[-2] for i in files])
        # lang = list(filter(lambda x: sum([ll in x for ll in limited_langs]) > 0, lang))
        languages = [
            "Amharic",
            # "Algerian Arabic",
            # "Morrocan Arabic",
            # "English",
            "Hausa",
            "Igbo",
            # "Kinyarwanda",
            # "Oromo",
            # "Nigerian Pidgin",
            # "Portuguese",
            "Swahili",
            # "Tigrinya",
            # "Tsonga",
            # "Twi",
            "Yoruba",
        ]
        return dict(zip(lang, languages))
    if mt:
        languages = [
            "Yoruba",
            # "Zulu",
            "Hausa",
            # "Setswana",
            "Swahili",
            # "Nigerian-Pidgin",
            # "Fon",
            # "Twi",
            # "Mossi",
            # "Ghomala",
            # "Wolof",
            # "Luganda",
            # "Chichewa",
            # "Bambara",
            # "Kinyarwanda",
            # "Luo",
            # "Ewe",
            # "Xhosa",
            "Igbo",
            "Amharic",
            # "Shona",
        ]
        lang = [i.split("/")[-2].split("-")[1] for i in files]
        # lang = list(filter(lambda x: sum([ll in x for ll in limited_langs]) > 0, lang))

        return dict(zip(lang, languages))


def sentiment(model_pipeline, tokenizer, output_dir):
    """Identifies tweet sentiments for different languages"""

    files = glob.glob(
        "data_repos/afrisent-semeval-2023//data/**/test.tsv", recursive=True
    )

    assert len(files) != 0

    files = list(filter(lambda x: sum([ll in x for ll in LIMITED_LANGS]) > 0, files))

    languages = get_language(files, senti=True)
    label = "{{Neutral, Positive or Negative}}"

    for file in tqdm(files):
        # language = file.split('/')[-2]
        # if language == 'eng':
        df = pd.read_csv(file, sep="\t", header=0)
        language = file.split("/")[-2]
        language = languages[language]
        # [v for k, v in languages.items() if k == language][0]
        print(f"\nLanguage: {language}, using file: {file}")
        print(df.head())
        responses = []
        for index in tqdm(range(df.shape[0])):
            text = df["tweet"].iloc[index]
            message = f'Does this {language} statement; "{text}" have a {label} sentiment? Labels only'
            input_mes = message + " "

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, "processed", result)

        completions = []
        for completion_text in responses:
            completions.append(completion_text)

        df["gpt2"] = completions
        df.to_csv(output_dir + language + ".tsv", sep="\t")


def news_classification(model_pipeline, tokenizer, output_dir):
    files = glob.glob("data_repos/masakhane-news/data/**/test.tsv", recursive=True)
    assert len(files) != 0
    files = list(filter(lambda x: sum([ll in x for ll in LIMITED_LANGS]) > 0, files))

    prompt_prefix = 'Is this a piece of news regarding {{"'
    prompt_suffix = '"}}? '

    for file in tqdm(files):
        file_path = Path(file)
        df = pd.read_csv(file, sep="\t")
        label_string, label_list = getlabel_string(
            Path(f"{file_path.parent}/labels.txt")
        )
        lang = file.split("/")[-2]
        print(f"\nLanguage: {lang}, using file: {file}")
        print(df.head())

        responses = []
        for index in tqdm(range(df.shape[0])):  # df.shape[0]
            headline = df["headline"].iloc[index]
            content = df["text"].iloc[index]
            text_string = headline + " " + content
            query = " ".join(text_string.split()[:100])

            message = (
                "Labels only. " + prompt_prefix + label_string + prompt_suffix + query
            )

            input_mes = message + " "

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, "processed", result)

        completions = []
        for completion_text in responses:
            completions.append(completion_text)

        df["gpt2"] = completions
        df.to_csv(output_dir + lang + ".tsv", sep="\t")


def cross_lingual_qa(model_pipeline, tokenizer, output_dir, pivot=False):
    languages = ["ibo", "yor", "hau", "swa"]
    for language in tqdm(languages):
        print(language)
        gold_passages = glob.glob(
            f"data_repos/afriqa/data/gold_passages/{language}/*test.json"
        )
        assert len(gold_passages) != 0
        gp_df = pd.read_json(gold_passages[0], lines=True)
        print(
            f"\nLanguage: {language}, using file: data_repos/afriqa/data/gold_passages/{language}/*test.json"
        )
        print(gp_df.head())

        pivot_lang = "French" if gold_passages[0].split(".")[-3] == "fr" else "English"
        prompt_query = f"Use the following pieces of context to answer the provided question. If you don't know the answer, \
just say that you don't know, don't try to make up an answer. Provide the answer with the least number of \
words possible. Provide the answer only. Provide answer in {pivot_lang}. Do not repeat the question"

        responses = []
        for index in tqdm(range(len(gp_df))):
            context = f"Context: {gp_df['context'].iloc[index]}"
            question = (
                f"Question: {gp_df['question_translated'].iloc[index]}"
                if pivot
                else f"Question: {gp_df['question_lang'].iloc[index]}"
            )
            message = prompt_query + "\n\n" + context + "\n" + question

            input_mes = message + " "

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, "processed", result)

        # print(responses[:10])
        completions = []
        for completion_text in responses:
            completions.append(completion_text)

        gp_df["gpt2"] = completions
        gp_df.to_csv(output_dir + language + ".tsv", sep="\t")


def machine_translation(model_pipeline, tokenizer, output_dir, reverse=False):
    files = glob.glob("data_repos/lafand-mt/data/tsv_files/**/test.tsv", recursive=True)
    assert len(files) != 0
    files = list(filter(lambda x: sum([ll in x for ll in LIMITED_LANGS]) > 0, files))

    languages = get_language(files, mt=True)

    for file in tqdm(files):
        df = pd.read_csv(file, sep="\t", header=0)
        pivot_lang_abv, target_lang_abv = (
            file.split("/")[-2].split("-")[0],
            file.split("/")[-2].split("-")[1],
        )
        target_lang = [v for k, v in languages.items() if k == target_lang_abv][0]
        pivot_lang = "English" if pivot_lang_abv == "en" else "French"

        print(f"\nLanguage: {target_lang}, using file: {file}")
        print(df.head())

        responses = []
        for index in tqdm(range(df.shape[0])):
            if not reverse:
                text = (
                    df["en"].iloc[index]
                    if pivot_lang_abv == "en"
                    else df["fr"].iloc[index]
                )
                prompt_query = f"Translate the {pivot_lang} sentence below to {target_lang}. Return the translated \
                sentence only. If you cannot translate the sentence simply say you don't know"
            else:
                text = df[target_lang_abv].iloc[index]
                prompt_query = f"Translate the {target_lang} sentence below to {pivot_lang}. Return the translated \
                                sentence only. If you cannot translate the sentence simply say you don't know"

            message = prompt_query + "\n" + text

            input_mes = message + " "

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, "processed", result)

        completions = []
        for completion_text in responses:
            completions.append(completion_text)
        df["gpt2"] = completions
        if reverse:
            df.to_csv(
                output_dir + f"{target_lang_abv}-{pivot_lang_abv}" + ".tsv", sep="\t"
            )
        else:
            df.to_csv(
                output_dir + f"{pivot_lang_abv}-{target_lang_abv}" + ".tsv", sep="\t"
            )


def named_entity_recognition(model_pipeline, tokenizer, output_dir):
    prompt_query = "Named entities refers to names of location, organisation and personal name. \n\
For example, 'David is an employee of Amazon and he is visiting New York next week to see Esther' will be \n\
PERSON: David $ ORGANIZATION: Amazon $ LOCATION: New York $ PERSON: Esther \n\n\
List all the named entities in the passage above using $ as separator. Return only the output"

    files = glob.glob(
        "data_repos/masakhane-ner/xtreme-up/MasakhaNER-X/test/*.jsonl", recursive=True
    )
    assert len(files) != 0

    files = list(filter(lambda x: sum([ll in x for ll in LIMITED_LANGS]) > 0, files))

    for file in tqdm(files):
        with open(file) as data:
            data_lines = data.read().splitlines()

        data_dicts = [json.loads(line) for line in data_lines]
        df = pd.DataFrame(data_dicts)
        df = df[~(df["target"] == "")]
        file_lang = file.split("/")[-1].split(".")[0]

        print(f"\nLanguage: {file_lang}, using file: {file}")
        print(df.head())

        responses = []
        for index in tqdm(range(df.shape[0])):
            text = df["text"].iloc[index]
            message = text + "\n\n" + prompt_query
            input_mes = message + " "

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, "processed", result)

        completions = []
        for completion_text in responses:
            completions.append(completion_text)

        df["gpt2"] = completions
        df.to_csv(output_dir + file_lang + ".tsv", sep="\t")


def main(
    senti: bool = False,
    news: bool = False,
    qa: bool = False,
    qah: bool = False,
    mt_from_en: bool = False,
    mt_to_en: bool = False,
    ner: bool = False,
):
    """Runs the task functions"""

    model_name = "gpt2"
    model_pipeline, tokenizer = load_model(model_name)

    if senti is True:
        output_dir = f"results_{model_name}/sentiment/"
        create_dir(output_dir)

        sentiment(model_pipeline, tokenizer, output_dir)
    elif news is True:
        output_dir = f"results_{model_name}/news_topic/"
        create_dir(output_dir)

        news_classification(model_pipeline, tokenizer, output_dir)
    elif qa is True:
        output_dir = f"results_{model_name}/qa/"
        create_dir(output_dir)

        cross_lingual_qa(model_pipeline, tokenizer, output_dir, pivot=True)
    elif qah is True:
        output_dir = f"results_{model_name}/qah/"
        create_dir(output_dir)

        cross_lingual_qa(model_pipeline, tokenizer, output_dir, pivot=False)
    elif mt_from_en is True:

        output_dir = f"results_{model_name}/mt/"
        create_dir(output_dir)

        machine_translation(model_pipeline, tokenizer, output_dir, reverse=False)
    elif mt_to_en is True:

        output_dir = f"results_{model_name}/mt/"
        create_dir(output_dir)

        machine_translation(model_pipeline, tokenizer, output_dir, reverse=True)
    elif ner is True:

        output_dir = f"results_{model_name}/ner/"
        create_dir(output_dir)

        named_entity_recognition(model_pipeline, tokenizer, output_dir)


if __name__ == "__main__":
    main(senti=True)
    main(news=True)
    main(qa=True)
    main(qah=True)
    main(mt_to_en=True)
    main(mt_from_en=True)
    main(ner=True)
