'''
Adapted from https://github.com/lupantech/ScienceQA
'''

from dataclasses import dataclass
from typing import List, Optional
from gensim.models import KeyedVectors

# 从本地加载模型
model = KeyedVectors.load("download/glove-wiki-gigaword-300.model")

def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption']
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt

def get_origin_answer(problem, options):
    return problem['choices'][problem['answer']]

def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True, WithOutput = False, curr_le_data=None):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'E':
            output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution}."

    elif output_format == 'E':
        output = f"Solution: {solution}"
        
    
    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        else: 
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output
        
        
    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text

import torch
def build_prompt(problems, shot_qids, test_qid, args):

    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input
def parse_triples(triples):
    parsed_triples = []

    for triple in triples:
        if len(triple) < 3:
            continue
        subject = triple[0]
        relation = triple[1]
        objects = triple[2:]
        for obj in objects:
            parsed_triples.append((subject, relation, obj))
    return parsed_triples
import numpy as np

def build_vocab(triples):
    nodes = set()
    relations = set()
    for subj, rel, obj in triples:
        nodes.update([subj, obj])
        relations.add(rel)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}
    return node_to_idx, relation_to_idx


def build_edges(triples, node_to_idx, relation_to_idx):
    edges = []
    edge_attrs = []
    for subj, rel, obj in triples:
        src = node_to_idx[subj]
        dst = node_to_idx[obj]
        edges.append((src, dst))
        edge_attrs.append(relation_to_idx[rel])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    return edge_index, edge_attr
class GraphData:
    def __init__(self, node_features, edge_index, edge_attr):
        self.x = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
def get_embedding(text, model, embedding_dim):
    words = text.lower().split()
    embeddings = [model[word] for word in words if word in model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)
def build_node_features(node_to_idx, model, embedding_dim):
    node_features = []
    for node in node_to_idx.keys():
        embedding = get_embedding(node, model, embedding_dim)
        node_features.append(embedding)
    node_features = torch.tensor(node_features, dtype=torch.float)
    return node_features

def build_train_pair(problems, test_qid, args, curr_le_data=None):

    examples = []

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])
    answer_option = get_answer(problems[test_qid], args.options)
    answer = "(" + answer_option + ")"
    
    test_example, target = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=False,WithOutput = True, curr_le_data=curr_le_data)
    examples.append(test_example)
    
    target = target.replace("Answer:", "").strip()
    if len(problems[test_qid]["triples"]) == 0:
        # node_to_idx, edge_index, edge_attr = None,None,None
        # graph = None
        # node_features = None
        data = None
    else:
        parsed_triples = parse_triples(problems[test_qid]["triples"])
        node_to_idx, relation_to_idx = build_vocab(parsed_triples)
        edge_index, edge_attr = build_edges(parsed_triples, node_to_idx, relation_to_idx)
        # graph = build_graph(parsed_triples, node_to_idx)
        def build_data_object(node_features, edge_index, edge_attr):
            # data = Data(node_features, edge_index, edge_attr)

            data = GraphData(node_features, edge_index=edge_index, edge_attr=edge_attr)
            return data

        # node_features = build_node_features(node_to_idx, model, 300)
        # graph.ndata['feat'] = node_features  # 设置节点特征

        node_features = build_node_features(node_to_idx, model, 300)
        data = build_data_object(node_features, edge_index, edge_attr)
        # print(data.x)
    # print(data.x.shape[1])
    if data.x is None:
        data = None
        # print("data.x is None")
        # data.x = torch.ones(1,300)
        # data.edge_index = torch.tensor([0,1])
    elif data.x.ndim < 2:
        data = None
        # print(f"data.x 的维度是 {data.x.ndim}，小于二维")
        # data.x = torch.ones(1,300)
        # data.edge_index = torch.tensor([0,1])
    elif data.x.shape[1]!=300:
        data = None
    elif torch.isnan(data.x).any():
        data = None
    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return data,prompt_input, target

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]