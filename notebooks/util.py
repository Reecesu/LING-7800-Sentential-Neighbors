import torch
import torchvision
import numpy as np
import pandas as pd
import altair as alt
import scipy.stats as stats
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def encode(df, tokenizer):

    embeddings = []

    for _, row in df.iterrows():

        # tokenize S1 and S2, pad to max length of 128, and truncate if needed
        encoded = tokenizer([row["S1"]], [row["S2"]], padding="max_length", truncation=True, max_length=128)
        
        # convert ids, attention mask, and labels to tensors (also, squeeze to remove extra dimension)
        input_ids = torch.tensor(encoded["input_ids"]).squeeze()
        attention_mask = torch.tensor(encoded["attention_mask"]).squeeze()
        label = torch.tensor(row["Level 1"]).long()

        # append the dictionary of tensors to our embeddings list
        embeddings.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": label})

    return embeddings

def evaluate(preds):

    # predictions should be a tuple, we can unpack it to logits and labels here
    logits, labels = preds

    # actual predictions are the argmax of the logits
    predictions = np.argmax(logits, axis=-1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'macro_f1': f1_score(labels, predictions, average='macro'),
        'f1': f1_score(labels, predictions, average='weighted'),
    }

# SOURCE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
# SOURCE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2996580/

def compare(metric, df1, df2):
    group_1 = df1[metric]
    group_2 = df2[metric]
    
    t_stat, p_value = stats.ttest_ind(group_1, group_2)
    
    return t_stat, p_value

def significance(metric, p_value, alpha=0.05):
    if p_value < alpha:
        return f"{metric}: P-value = {p_value} (Significant!)"
    else:
        return f"{metric}: P-value = {p_value} (Insignificant!)"


def create_df(paths):

    my_data = []
    columns = {

        'eval_loss': 'loss',
        'eval_accuracy': 'accuracy',
        'eval_precision': 'precision',
        'eval_recall': 'recall',
        'eval_macro_f1': 'macro f1',
        'eval_f1': 'f1'

    }

    for i in paths:

        df = pd.read_csv(i)
        df = df[df['eval_loss'].notnull()]
        model = i.replace('../results/', '').replace('.csv', '')
        df['model'] = model
        my_data.append(df)

    df2 = pd.concat(my_data, axis=0).reset_index(drop=True)
    df2.rename(columns=columns, inplace=True)
    df2 = df2[['model', 'epoch', 'accuracy', 'precision', 'recall', 'macro f1', 'f1']]

    return df2

# SOURCE: https://altair-viz.github.io/gallery/line_chart_with_custom_legend.html

def static_graph(df, metric):
    base = alt.Chart(df).encode(
        color=alt.Color("model", legend=alt.Legend(title="Model"))
    ).properties(
        width=500
    )

    line = base.mark_line().encode(
        x="epoch:Q",
        y=f"{metric}:Q",
        tooltip=["model:N", "epoch:Q", f"{metric}:Q"]
    )

    last_epoch = base.mark_circle().encode(
        x=alt.X("last_epoch['epoch']:Q"),
        y=alt.Y(f"last_epoch['{metric}']:Q")
    ).transform_filter(
        alt.datum.metric == metric
    ).transform_aggregate(
        last_epoch="argmax(epoch)",
        groupby=["model"]
    )

    model_name = last_epoch.mark_text(align="left", dx=4).encode(text="model")

    chart = (line + last_epoch + model_name).encode(
        x=alt.X(title="Epoch"),
        y=alt.Y(title=metric.capitalize())
    ).properties(
        title=f"{metric.capitalize()} per Epoch"
    ).interactive()

    return chart

# SOURCE: https://altair-viz.github.io/gallery/line_chart_with_custom_legend.html
# SOURCE: https://altair-viz.github.io/gallery/multiple_interactions.html

def dynamic_graph(df):
    melt = pd.melt(df, id_vars=["model", "epoch"], var_name="metric", value_name="value")
    selection = alt.selection_single(
        name="Select",
        fields=["metric"],
        init={"metric": "accuracy"},
        bind=alt.binding_select(options=["accuracy", "precision", "recall", "macro f1", "f1"])
    )

    base = alt.Chart(melt).encode(
        color=alt.Color("model", legend=alt.Legend(title="Model"))
    ).properties(
        width=500
    )

    line = base.mark_line().encode(
        x="epoch:Q",
        y="value:Q",
        tooltip=["model:N", "epoch:Q", "value:Q", "metric:N"]
    ).add_selection(
        selection
    ).transform_filter(
        selection
    )

    last_epoch = base.mark_circle().encode(
        x=alt.X("last_epoch['epoch']:Q"),
        y=alt.Y("last_epoch['value']:Q")
    ).transform_aggregate(
        last_epoch="argmax(epoch)",
        groupby=["model"]
    ).transform_filter(
        selection
    )

    chart = (line + last_epoch).encode(
        x=alt.X(title="Epoch"),
        y=alt.Y("value:Q", title="Value"),
    ).properties(
        title="Evaluation Metrics per Epoch"
    ).add_selection(
        selection
    ).interactive()

    return chart

def EWN(df, n=1):

    updated_S1 = []
    updated_S2 = []

    for i in range(len(df)):


        FileNumber = df.loc[i, 'FileNumber']
        SectionNumber = df.loc[i, 'SectionNumber']
        previous_S1 = []
        next_S2 = []

        # append all previous senteces to previous_S1 (given the same file number and section number)
        for j in range(1, n + 1):
            if (i - j >= 0) and (df.loc[i - j, 'FileNumber'] == FileNumber) and (df.loc[i - j, 'SectionNumber'] == SectionNumber):
                previous_S1.append(df.loc[i - j, 'S2'])

        # append all next seneteces to next_S2 (given the same file number and section number)
        for j in range(1, n + 1):
            if (i + j < len(df)) and (df.loc[i + j, 'FileNumber'] == FileNumber) and (df.loc[i + j, 'SectionNumber'] == SectionNumber):
                next_S2.append(df.loc[i + j, 'S1'])

        # if there are previous or next sentences, join them with the current sentence
        if previous_S1:
            updated_S1.append(' '.join(previous_S1) + ' ' + df.loc[i, 'S1'])
        else:
            updated_S1.append(df.loc[i, 'S1'])

        if next_S2:
            updated_S2.append(df.loc[i, 'S2'] + ' ' + ' '.join(next_S2))
        else:
            updated_S2.append(df.loc[i, 'S2'])

    df2 = df.copy()

    df2['S1'] = updated_S1
    df2['S2'] = updated_S2

    return df2

def PSRN(df, n=1):

    def get_random_sentence(section_number, file_number, num_sentences, index):
        neighbors = shuffle.loc[
            (shuffle['SectionNumber'] == section_number) &
            (shuffle['FileNumber'] == file_number) &
            (shuffle.index != index)
        ].sample(num_sentences)

        return neighbors['S1'].tolist(), neighbors['S2'].tolist()
    
    # Shuffle the dataframe to get random sentence orders
    shuffle = df.sample(frac=1).reset_index(drop=True)

    # copy df to join sentences
    df2 = df.copy()

    for idx, row in df.iterrows():

        # Get random sentences with the same FileNumber and SectionNumber
        previous_S1, next_S2 = get_random_sentence(row['SectionNumber'], row['FileNumber'], n, idx)

        # Prepend random sentences to S1 and append random sentences to S2
        df2.at[idx, 'S1'] = ' '.join(previous_S1 + [row['S1']])
        df2.at[idx, 'S2'] = ' '.join([row['S2']] + next_S2/2)

    return df2

def direct_neighbors(df, n=1):

    updated_S1 = []
    updated_S2 = []
    previous_S1 = []
    next_S2 = []

    for i in range(len(df)):


        SectionNumber = df.loc[i, 'SectionNumber']
        FileNumber = df.loc[i, 'FileNumber']
        SentenceNumber = int(df.loc[i, 'SentenceNumber'])


        # get previous sentences (given the same file number and section number)
        for j in range(1, n + 1):
            if (i - j >= 0 and df.loc[i - j, 'SectionNumber'] == SectionNumber and
                    df.loc[i - j, 'FileNumber'] == FileNumber and
                    int(df.loc[i - j, 'SentenceNumber']) == SentenceNumber - j):
                previous_S1.append(df.loc[i - j, 'S2'])

        # get next sentences (given the same file number and section number)
        for j in range(1, n + 1):
            if (i + j < len(df) and df.loc[i + j, 'SectionNumber'] == SectionNumber and
                    df.loc[i + j, 'FileNumber'] == FileNumber and
                    int(df.loc[i + j, 'SentenceNumber']) == SentenceNumber + j):
                next_S2.append(df.loc[i + j, 'S1'])


        # filter duplicates
        if previous_S1 and previous_S1[-1] == df.loc[i, 'S1']:
            previous_S1.pop()

        if next_S2 and next_S2[0] == df.loc[i, 'S2']:
            next_S2.pop(0)


        # add <s1> and <s2> tags
        previous_S1 = ['<s1>' + s + '</s1>' for s in previous_S1]
        next_S2 = ['<s2>' + s + '</s2>' for s in next_S2]

        # concatenate the sentences
        updated_S1.append(' '.join(previous_S1 + ['<s1>' + df.loc[i, 'S1'] + '</s1>']))
        updated_S2.append(' '.join(['<s2>' + df.loc[i, 'S2'] + '</s2>'] + next_S2))

    df2 = df.copy()

    # add udated sentences to the dataframe
    df2['S1'] = updated_S1
    df2['S2'] = updated_S2

    # filter rows where no expanded window is added (we want at least 1 previous or next sentence)
    # NOTE: Change the | operator to & if you want to only include double neighbors
    filtered_df = df2[(df2['S1'] != '<s1>' + df['S1'] + '</s1>') | (df2['S2'] != '<s2>' + df['S2'] + '</s2>')]

    return filtered_df