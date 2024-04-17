import glob
import json
import argparse
import pandas as pd
from sklearn.metrics import f1_score

import utils_gpt2


def senti_eval(prediction_files, metric):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t")
        df["gpt2_split"] = df["gpt2"].str.lower()
        df["gpt2_split"] = df["gpt2_split"].str.replace("\n", " ", regex=True)
        df["gpt2_split"] = df["gpt2_split"].str.replace(
            r"([^\w\s{{}}])", "", regex=True
        )
        df.fillna("unknown", inplace=True)
        df["gpt2_split"] = df["gpt2_split"].apply(utils_gpt2.normalize_senti_text)

        print(f"Example sentiment generations: {str(df["gpt2_split"][0])}")

        lang = file.split("/")[-1].split(".")[0]
        df["gpt2_label"] = df.apply(
            utils_gpt2.gpt2_extract_senti_label, axis=1, args=(lang,)
        )
        df = utils_gpt2.filter_senti_labels(df)

        f1 = round(f1_score(df["label"], df["gpt2_label"], average="weighted") * 100, 2)

        language = utils_gpt2.language_abv(lang)
        metric[language] = f1
    return metric


def ner_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t", header=0)
        df["gpt2"] = df["gpt2"].str.lower()
        df["gpt2"] = df["gpt2"].str.replace("\n", " ", regex=True)
        df["gpt2"] = df["gpt2"].str.replace("</s>", "", regex=True)
        df["gpt2"] = df["gpt2"].str.split("please").str[0].str.strip()
        df["gpt2"] = df["gpt2"].str.split("i hope").str[0].str.strip()

        df["gpt2"] = df.apply(utils_gpt2.gpt2_extract_ner_pred, axis=1)

        df["target"] = df["target"].apply(utils_gpt2.format_ner_text, target=True)
        df["gpt2"] = df["gpt2"].apply(utils_gpt2.format_ner_text, target=False)
        df = df[~(df.target == "")]

        print(f"Example ner generations: {df["gpt2"].head()}")

        f1 = utils_gpt2.calculate_ner_metrics(df, "gpt2")
        language = file.split("/")[-1].split(".")[0]
        metrics[language] = f1
    return metrics


def mt_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t")
        df["gpt2"] = df["gpt2"].str.lower()
        df["gpt2_split"] = df["gpt2"].str.replace("\n\n", " ", regex=False)
        df["gpt2_split"] = df["gpt2_split"].str.replace("\n", " ", regex=False)
        df["gpt2_split"] = df["gpt2_split"].str.replace("</s>", " ", regex=False)
        df["gpt2_split"] = (
            df["gpt2_split"].str.split("with that said").str[-1].str.strip()
        )
        df["gpt2_split"] = (
            df["gpt2_split"]
            .str.split("with those limitations in mind")
            .str[-1]
            .str.strip()
        )
        df["gpt2_split"] = (
            df["gpt2_split"]
            .str.split("with those considerations in mind")
            .str[-1]
            .str.strip()
        )

        print(f"Example mt generations: {str(df["gpt2_split"][0])}")

        lang_full = file.split("/")[-1].split(".")[0]

        if lang_full.split("-")[1] == "eng":
            lang = "eng_Latn"
            language = "english"
        elif lang_full.split("-")[1] == "fra":
            lang = "fra_Latn"
            language = "french"
        elif lang_full.split("-")[1] == "deu":
            lang = "deu_Latn"
            language = "german"
        else:
            lang = lang_full.split("-")[1]
            language = utils_gpt2.lang_dict[lang].lower()

        df["gpt2_reponse"] = df.apply(
            utils_gpt2.gpt2_extract_mt_pred, axis=1, args=(language,)
        )
        df["gpt2_reponse"] = (
            df["gpt2_reponse"].str.split("i hope this helps").str[0].str.strip()
        )
        df["gpt2_reponse"] = (
            df["gpt2_reponse"].str.split("i hope that helps").str[0].str.strip()
        )
        df["gpt2_reponse"] = (
            df["gpt2_reponse"].str.split("please note that").str[0].str.strip()
        )

        df[[lang, "gpt2_reponse"]] = df[[lang, "gpt2_reponse"]].applymap(
            utils_gpt2.normalize_text
        )

        lang_metric = utils_gpt2.calculate_mt_metrics(df, "gpt2_reponse", lang)
        metrics[lang_full] = lang_metric
    return metrics


def qa_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t")
        df["translated_answer"] = df["answer_pivot"].apply(
            lambda x: x.split(": ")[-1].strip("['").rstrip("']}")
        )
        df["gpt2_response"] = df["gpt2"].str.lower()
        # df["gpt2_response"] = (
        #     df["gpt2_response"].str.split("information provided,").str[-1].str.strip()
        # )
        # df["gpt2_response"] = (
        #     df["gpt2_response"].str.split("information provided").str[-1].str.strip()
        # )
        df["gpt2_response"] = (
            df["gpt2_response"].str.split("answer :").str[-1].str.strip()
        )
        # df["gpt2_response"] = df["gpt2_response"].str.split("\n").str[-1].str.strip()
        df["gpt2_response"] = df["gpt2_response"].str.replace("\n", " ", regex=True)
        df["gpt2_response"] = df["gpt2_response"].str.replace("</s>", "", regex=False)

        df[["gpt2_response", "translated_answer"]] = df[
            ["gpt2_response", "translated_answer"]
        ].applymap(utils_gpt2.normalize_text)
        df = df[~(df["translated_answer"] == "")]

        print(f"Example qa generations: {str(df["gpt2_response"][0])}")

        df["gpt2_response"] = df.apply(utils_gpt2.check_yes_no, axis=1)

        language = file.split("/")[-1].split(".")[-2]

        lang_metric = utils_gpt2.calculate_qa_metrics(df, "gpt2_response")

        metrics[language] = lang_metric
    return metrics


def news_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t")
        df["gpt2"] = df["gpt2"].str.lower()
        df["gpt2_split"] = df["gpt2"].str.replace("\n\n", " ", regex=False)
        df["gpt2_split"] = df["gpt2_split"].str.replace("\n", " ", regex=False)
        df["gpt2_split"] = df["gpt2_split"].str.replace("</s>", "", regex=False)
        df["gpt2_split"].fillna("", inplace=True)

        print(f"Example news generations: {str(df["gpt2_split"][0])}")

        df["gpt2_label"] = df.apply(utils_gpt2.gpt2_extract_news_label, axis=1)
        df[["category", "gpt2_label"]] = df[["category", "gpt2_label"]].applymap(
            utils_gpt2.normalize_text
        )

        # if it contains more than one label
        df["gpt2_label"] = df["gpt2_label"].apply(
            lambda x: "unknown" if x.count(" ") >= 1 else x
        )

        # assign random labels to unknowns
        df["gpt2_label"] = df.apply(
            utils_gpt2.assign_label, axis=1, row_name="gpt2_label"
        )

        f1 = round(
            (f1_score(df["category"], df["gpt2_label"], average="weighted") * 100), 2
        )

        language = file.split("/")[-1].replace(".tsv", "")
        metrics[language] = f1
    return metrics


def main(args, prediction_dir, task_function):
    args, unknown = args
    output_dir = args.output_directory

    prediction_files = glob.glob(f"{prediction_dir}/*tsv", recursive=True)
    metrics = {}

    results = task_function(prediction_files, metrics)
    utils_gpt2.create_dir(output_dir)
    task_name = prediction_dir.split("/")[-1]
    with open(f"{output_dir}/{task_name}", "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_directory", type=str, default="../results/gpt2/")

    args = parser.parse_known_args()

    print("QA Eval...")
    main(args, "../predictions/results_gpt2/qa", qa_eval)

    print("QAh Eval...")
    main(args, "../predictions/results_gpt2/qah", qa_eval)

    print("MT Eval...")
    main(args, "../predictions/results_gpt2/mt", mt_eval)

    print("NER Eval...")
    main(args, "../predictions/results_gpt2/ner", ner_eval)

    print("News topic Eval...")
    main(args, "../predictions/results_gpt2/news_topic", news_eval)

    print("Sentiment Eval...")
    main(args, "../predictions/results_gpt2/sentiment", senti_eval)
