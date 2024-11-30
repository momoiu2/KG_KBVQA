from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from kg_utils_okvqa_prompt import *

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
}


def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val': val_qids, 'test': test_qids}
    return problems, qids,


def load_okvqa_data_img(args):
    # args.data_root = "C:\program_soft\pythoncode\mm-cot-main\mm-cot-main\data\ok_vqa"
    train_problems = json.load(open(os.path.join(args.data_root, 'ok_vqa/ok_vqa_train_new_new1.json')))  # 问题
    test_problems = json.load(open(os.path.join(args.data_root, 'ok_vqa/ok_vqa_test_new_new2.json')))
    val_problems = json.load(open(os.path.join(args.data_root, 'ok_vqa/ok_vqa_train_new_new.json')))
    # train_problems = json.load(open('ok_vqa_train_new_new.json'))  # 问题
    # test_problems = json.load(open('ok_vqa_train_new_new.json'))
    # val_problems = json.load(open('ok_vqa_train_new_new.json'))
    train_name_maps = json.load(open('data/ok_vqa/vision_features'
                                     '/train_name_map.json'))
    test_name_maps = json.load(open('data/ok_vqa/vision_features'
                                    '/test_name_map.json'))
    val_name_maps = json.load(open('data/ok_vqa/vision_features'
                                   '/train_name_map.json'))
    start = args.start
    end = args.end
    keys = list(test_problems.keys())
    test_problems = {key: test_problems[key] for key in keys[start:end]}
    test_json_name = args.test_json_name

    # check
    # 图片所用特征 np加载npy文件

    train_qids = []
    train_qids1 = train_problems.keys()
    for q_id in train_qids1:
        train_qids.append(q_id)

    test_qids = []
    test_qids1 = test_problems.keys()
    for q_id in test_qids1:
        test_qids.append(q_id)

    val_qids = []
    val_qids1 = val_problems.keys()
    for q_id in val_qids1:
        val_qids.append(q_id)
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")
    if args.img_type == "resnet":
        image_features = np.load('vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        train_image_features = np.load(
            'data/ok_vqa/vision_features/okvqa_clip_train.npy')
        val_image_features = np.load(
            'data/ok_vqa/vision_features/okvqa_clip_test.npy')
        test_image_features = np.load(
            'data/ok_vqa/vision_features/okvqa_clip_train.npy')
    elif args.img_type == "detr":
        train_image_features = np.load('data/ok_vqa/vision_features/okvqa_detr_train.npy')
        val_image_features = np.load('data/ok_vqa/vision_features/okvqa_detr_val.npy')
        test_image_features = np.load('data/ok_vqa/vision_features/okvqa_detr_test.npy')
    else:
        image_features = np.load('vision_features/detr.npy')

    qids = {'train': train_qids, 'val': val_qids, 'test': test_qids}

    problems = {'train_problems': train_problems, 'test_problems': test_problems, 'val_problems': val_problems}
    name_maps = {'train_name_maps': train_name_maps, 'val_name_maps': val_name_maps, 'test_name_maps': test_name_maps}
    return test_json_name,problems, qids, name_maps, train_image_features, val_image_features, test_image_features


class OKVQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    创建自定义数据集以读取数据集和
    将其加载到数据加载器以将其传递给
    用于模型微调的神经网络
    """

    def __init__(
            self, embedding_layer, t_t_v, problems, qids, name_maps, tokenizer, source_len, target_len, args,
            image_features,
            test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
            source_len（int）：源文本的最大长度

            target_len（int）：目标文本的最大长度
            source_text（str）：源文本的列名
            target_text（str）：目标文本的列名

        """
        self.tokenizer = tokenizer
        self.data = problems  # 遍历所有的问题，以列表方式存在
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []  # 图片的向量
        self.edge_index = []
        self.node_to_idx = []
        self.edge_attr = []
        self.data = []
        self.features = []
        if test_le is not None:  # 测试用到的
            test_le_data = json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in qids:
            if len(problems[qid]["triples"]) == 0:
                continue
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            data, prompt, target = build_train_pair(embedding_layer, t_t_v, problems, qid, args,
                                                    curr_le_data)
            if data is None:
                continue
            self.target_text.append(target)
            self.source_text.append(prompt)
            self.data.append(data)
            # self.features.append(feature)
            # self.edge_attr.append(edge_attr)
            # self.node_to_idx.append(node_to_idx)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)



    def __getitem__(self, index):
        """return the input ids, attention masks, target ids, and data (graph)"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]
        data = self.data[index]  # Data object from torch_geometric.data
        # data = dgl.to_homogeneous(data)
        # feature = self.features[index]
        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        # 提取 Data 对象的属性
        x = torch.tensor(data.x.clone().detach()) if data is not None else torch.ones(3,300)
        edge_index = torch.tensor(data.edge_index.clone().detach()) if data is not None else torch.tensor([0,1])
        edge_attr = torch.tensor(data.edge_attr.clone().detach()) if data is not None else torch.tensor([[1, 0],
        [0, 2]])

        image_ids = torch.tensor(image_ids).squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
            # "graph": data,  # 节点特征
            # "features":feature,
            "data":x,
            "edge_index": edge_index,  # 边索引
            "edge_attr": edge_attr,  # 边属性
        }

