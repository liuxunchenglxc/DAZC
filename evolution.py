import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from lib.datasets import build_dataset
from lib import utils
from supernet_engine import evaluate
from model.supernet_transformer import Vision_TransformerSuper
import argparse
import os
import yaml
from lib.config import cfg, update_config_from_file

import json
import torchvision
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from scipy.stats import kendalltau
from timm.models import create_model
from timm.data import resolve_model_data_config, create_transform
from lib.datasets import build_transform
from PIL import Image

import math
import copy

def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), cand_tuple[-1]

class EvolutionSearcher(object):

    def __init__(self, args, device, model, model_without_ddp, choices, test_loader, output_dir, data_path, transforms, kendall, teacher_model):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.s_prob =args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices

        self.data_path = data_path
        self.transforms = transforms
        self.kendall = kendall
        self.teacher_model = teacher_model
        self.eval_test = True
        self.infer_teacher = True

        self.zc_imgs = [] # zero cost with special inputs
        self.zc_labels = [] # labels of special inputs
        self.t_imgs = [] # inputs for teacher model
        self.nb_classes = args.nb_classes # number of classes
        self.proxy = args.measure_name # proxy measure name
        self.data_load = args.data_load # data load method of minibatch

    def save_checkpoint(self, path=None):
        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        if path:
            checkpoint_path = path
        else:
            checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)

    def load_checkpoint(self, path=""):
        if not os.path.exists(self.checkpoint_path) and not os.path.exists(path):
            return False
        if path:
            info = torch.load(path)
        else:
            info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if self.kendall and not self.eval_test:
            pass
        else:
            if 'visited' in info:
                return False
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
        sampled_config = {}
        sampled_config['layer_num'] = depth
        sampled_config['mlp_ratio'] = mlp_ratio
        sampled_config['num_heads'] = num_heads
        sampled_config['embed_dim'] = [embed_dim]*depth
        with torch.no_grad():
            n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
            flops = self.model_without_ddp.get_complexity(sequence_length=196)
        info['params'] =  n_parameters / 10.**6
        info['flops'] =  flops / 10.**9

        if info['params'] > self.parameters_limits:
            print('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            print('under minimum parameters limit')
            return False

        if self.infer_teacher:
            # Calc teacher things
            if self.args.amp:
                with torch.cuda.amp.autocast():
                    self.teacher_output = self.teacher_model(self.t_imgs)
            else:
                self.teacher_output = self.teacher_model(self.t_imgs)
            self.infer_teacher = False
        
        # when small val_loader is used, test_loader should be replaced
        eval_stats = evaluate(self.test_loader, self.model, self.device, 
                              (self.data_load, 1, self.nb_classes), self.zc_imgs, self.zc_labels, self.teacher_output, self.proxy,
                              amp=self.args.amp, mode='retrain', retrain_config=sampled_config)
        info['acc'] = eval_stats['acc1']
        
        if self.eval_test:
            test_stats = evaluate(self.test_loader, self.model, self.device, amp=self.args.amp, mode='retrain', retrain_config=sampled_config)
            info['test_acc'] = test_stats['acc1']
            info['test_acc5'] = test_stats['acc5']
            

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):

        cand_tuple = list()
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(self.choices['depth'])
        cand_tuple.append(depth)
        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.choices[dimension]))

        cand_tuple.append(random.choice(self.choices['embed_dim']))
        return tuple(cand_tuple)

    def get_random(self, num):
        # only calc kendall and not eval test
        if self.kendall and not self.eval_test:
            self.infer_teacher = True
            for cand in self.candidates:
                self.is_legal(cand)
            return
        # normal progress
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            random_s = random.random()

            # depth
            if random_s < s_prob:
                new_depth = random.choice(self.choices['depth'])

                if new_depth > depth:
                    mlp_ratio = mlp_ratio + [random.choice(self.choices['mlp_ratio']) for _ in range(new_depth - depth)]
                    num_heads = num_heads + [random.choice(self.choices['num_heads']) for _ in range(new_depth - depth)]
                else:
                    mlp_ratio = mlp_ratio[:new_depth]
                    num_heads = num_heads[:new_depth]

                depth = new_depth
            # mlp_ratio
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    mlp_ratio[i] = random.choice(self.choices['mlp_ratio'])

            # num_heads

            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    num_heads[i] = random.choice(self.choices['num_heads'])

            # embed_dim
            random_s = random.random()
            if random_s < s_prob:
                embed_dim = random.choice(self.choices['embed_dim'])

            result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]

            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def calc_kendall(self, image_paths):
        zc_imgs = []
        zc_labels = []
        t_imgs = []
        class_to_idx = self.test_loader.dataset.class_to_idx
        for one_file in image_paths:
            origin_class = one_file.split('/')[-2]
            zc_label = class_to_idx[origin_class]
            one_img = Image.open(one_file)
            if one_img.mode != 'RGB':
                one_img = one_img.convert('RGB')
            zc_img = self.transforms[0](one_img)
            t_img = self.transforms[1](one_img)
            zc_imgs.append(zc_img)
            zc_labels.append(zc_label)
            t_imgs.append(t_img)
        self.zc_imgs = torch.stack(zc_imgs, dim=0)
        self.zc_labels = torch.as_tensor(zc_labels, dtype=torch.int32).type(torch.LongTensor)
        self.t_imgs = torch.stack(t_imgs, dim=0).to(torch.float16)
        self.get_random(100)
        score = []
        test_acc = []
        for cand in self.candidates:
            info = self.vis_dict[cand]
            score.append(info['acc'])
            test_acc.append(info['test_acc'])
        coef, p_value = kendalltau(score, test_acc)
        return coef, p_value
    
    def eval_kendall_individuals(self, individuals, file_list):
        # interp image paths
        for i in range(len(individuals)):
            if "files" in individuals[i]: # eval-ed
                continue
            files = []
            for unit, clss_files in zip(individuals[i]["gene"], file_list):
                files.append(clss_files[unit["value"]])
            individuals[i]["files"] = [i for i in files]
            coef, p_value = self.calc_kendall(files)
            individuals[i]["coef"] = coef
            individuals[i]["p_value"] = p_value
        return individuals

    def evo_kendall_select(self, individuals):
        i_1 = random.choice(individuals)
        i_2 = random.choice(individuals)
        if abs(i_1["coef"]) > abs(i_2["coef"]):
            return i_1
        else:
            return i_2
    
    def evo_kendall_crossover(self, p_1, p_2, l, n):
        # p_1 and p_2 are symmetric from select
        # q(n) = n ** -2
        if random.random() > n ** -2:
            return copy.deepcopy(p_1)
        # one point crossover
        pos = random.randint(0, l - 1)
        c = False
        res_gene = []
        for u_1, u_2 in zip(p_1["gene"], p_2["gene"]):
            if u_1["len"] <= pos: # not on this unit
                pos -= u_1["len"]
                res_gene.append(copy.deepcopy(u_1))
            else:
                if c: # crossovered
                    res_gene.append(copy.deepcopy(u_2))
                else: # pos < u_1["len"], pos is this index
                    mask = (1 << pos) - 1
                    value = u_1["value"] & mask
                    value += (u_2["value"] >> pos) << pos
                    new_u = copy.deepcopy(u_1)
                    new_u["value"] = value % (new_u["max"] + 1)
                    res_gene.append(new_u)
                    c = True
        return {"gene": res_gene}
    
    def evo_kendall_mutant(self, individual, l, n):
        # p(n) = n ** (-1.0/l)
        if random.random() > n ** (-1.0 / l):
            return copy.deepcopy(individual)
        # choice one position to mutant
        Allow = False
        res_gene = []
        while not Allow:
            pos = random.randint(0, l - 1)
            c = False
            res_gene = []
            for u in individual["gene"]:
                if u["len"] <= pos: # not on this unit
                    pos -= u["len"]
                    res_gene.append(copy.deepcopy(u))
                else:
                    if c: # mutanted
                        res_gene.append(copy.deepcopy(u))
                    else: # pos < u["len"], pos is this index
                        reverse = 1 << pos
                        value = u["value"] ^ reverse
                        if value > u["max"]: # not available for path
                            break
                        new_u = copy.deepcopy(u)
                        new_u["value"] = value
                        res_gene.append(new_u)
                        c = True
                        Allow = True
        return {"gene": res_gene}

    def evo_kendall_individuals(self, individuals, K, l, n):
        # find best one to keep
        best = individuals[0]
        for individual in individuals:
            if abs(individual["coef"]) > abs(best["coef"]):
                best = individual
        print("best tau is", abs(best["coef"]))
        # select parents to crossover
        crossover_res = []
        for _ in range(K-1):
            p_1 = self.evo_kendall_select(individuals)
            p_2 = self.evo_kendall_select(individuals)
            crossover_res.append(self.evo_kendall_crossover(p_1, p_2, l, n))
        # mutant the crossover_res to obtain offspring
        mutant_res = []
        for i in crossover_res:
            mutant_res.append(self.evo_kendall_mutant(i, l, n))
        return [copy.deepcopy(best)] + mutant_res

    def kendall_search(self, N=20, cluster_json=""):
        print(f"Start Kendall Search, N = {N}")
        # Cluster impl
        with open(cluster_json, "r") as f:
            cls_cluster = json.load(f)
        # form the file list
        file_list = []
        for clss in cls_cluster:
            # find all files of clss
            clss_files = []
            for cls in clss:
                directory = os.path.join(self.data_path, cls)
                files = []
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        files.append(filepath)
                # sample the dataset on each cls
                files = random.sample(files, 50)
                clss_files += files
            file_list.append(clss_files)
        print("Loaded Image Path List")
        # form the 01 gene based on the number of files store in int type
        base_gene = []
        for files in file_list:
            gene_unit = {"max": len(files) - 1, "value": 0, "len": math.ceil(math.log2(len(files)))}
            base_gene.append(gene_unit)
        # K = 2 * l
        l = sum([unit["len"] for unit in base_gene])
        K = 2 * l
        print(f"The gene's length is {l}, and K is {K}.")
        # init random genes
        individuals = []
        for _ in range(int(K)):
            gene = [{"max": unit["max"], "value": random.randint(0, unit["max"]), "len": unit["len"]} for unit in base_gene]
            individuals.append({"gene": gene})
        # eval these individuals
        individuals = self.eval_kendall_individuals(individuals, file_list)
        print("Init generation is ok")
        # record individuals
        record = []
        record.append(copy.deepcopy(individuals))
        # evo loop
        for i in range(N):
            # n-th generation
            n = i + 1
            # evo these individuals
            individuals = self.evo_kendall_individuals(individuals, K, l, n)
            # eval these individuals
            individuals = self.eval_kendall_individuals(individuals, file_list)
            # record individuals
            record.append(copy.deepcopy(individuals))
            print(f"{n}-th generation is ok")
        # print best
        best = individuals[0]
        for individual in individuals:
            if abs(individual["coef"]) > abs(best["coef"]):
                best = individual
        print("best tau is", abs(best["coef"]))
        return record

    def search(self, record_json=None):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        self.eval_test = False

        zc_imgs = []
        zc_labels = []
        t_imgs = []
        class_to_idx = self.test_loader.dataset.class_to_idx
        coef_sign = 1
        if record_json:
            with open(record_json, "r") as f:
                record = json.load(f)
                paths = max(record[-1], key=lambda x: abs(x["coef"]))["files"]
                max_coef = max(record[-1], key=lambda x: abs(x["coef"]))["coef"]
                assert max_coef != 0, 'cannot let all coefficients be 0'
                coef_sign = max_coef / abs(max_coef)
                for one_file in paths:
                    origin_class = one_file.split('/')[-2]
                    zc_label = class_to_idx[origin_class]
                    one_img = Image.open(one_file)
                    if one_img.mode != 'RGB':
                        one_img = one_img.convert('RGB')
                    zc_img = self.transforms[0](one_img)
                    t_img = self.transforms[1](one_img)
                    zc_imgs.append(zc_img)
                    zc_labels.append(zc_label)
                    t_imgs.append(t_img)
        else:
            #load random image and calc zero-cost score
            is_print_num = False
            is_cluster = True
            if is_cluster:
                # the clusters
                sum_num = 1.0
                with open("cls_cluster_5.json", "r") as f:
                    cls_cluster = json.load(f)
                for clss in cls_cluster:
                    cls = random.choice(clss)
                    directory = os.path.join(self.data_path, cls)
                    files = []
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        if os.path.isfile(filepath):
                            files.append(filepath)
                    one_file = random.choice(files)
                    origin_class = one_file.split('/')[-2]
                    zc_label = class_to_idx[origin_class]
                    print(one_file)
                    one_img = Image.open(one_file)
                    if one_img.mode != 'RGB':
                        one_img = one_img.convert('RGB')
                    zc_img = self.transforms[0](one_img)
                    t_img = self.transforms[1](one_img)
                    zc_imgs.append(zc_img)
                    zc_labels.append(zc_label)
                    t_imgs.append(t_img)
                    if is_print_num:
                        sum_num *= len(files)
                if is_print_num:
                    print("The image search space size:", sum_num)
            else:
                # non clusters
                num = 5
                # find all dirs
                subdirectories = []
                for entry in os.scandir(self.data_path):
                    if entry.is_dir():
                        subdirectories.append(entry.path)
                # find all files
                files = []
                for directory in subdirectories:
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        if os.path.isfile(filepath):
                            files.append(filepath)
                for _ in range(num):
                    one_file = random.choice(files)
                    origin_class = one_file.split('/')[-2]
                    zc_label = class_to_idx[origin_class]
                    print(one_file)
                    one_img = Image.open(one_file)
                    if one_img.mode != 'RGB':
                        one_img = one_img.convert('RGB')
                    zc_img = self.transforms[0](one_img)
                    t_img = self.transforms[1](one_img)
                    zc_imgs.append(zc_img)
                    zc_labels.append(zc_label)
                    t_imgs.append(t_img)
                if is_print_num:
                    print("The image search space size:", math.comb(len(files), num))
        # image res    
        self.zc_imgs = torch.stack(zc_imgs, dim=0)
        self.zc_labels = torch.as_tensor(zc_labels, dtype=torch.int32).type(torch.LongTensor)
        self.t_imgs = torch.stack(t_imgs, dim=0).to(torch.float16)

        # Calc Kendall
        if self.kendall:
            self.eval_test = True
            self.get_random(100)
            score = []
            test_acc = []
            for cand in self.candidates:
                info = self.vis_dict[cand]
                score.append(info['acc'])
                test_acc.append(info['test_acc'])
            coef, p_value = kendalltau(score, test_acc)
            print("Kendall tau: ", coef)
            print("p: ", p_value)
            return coef, p_value

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'] * coef_sign)
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'] * coef_sign)

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 val acc = {}, Top-1 test acc = #, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['params']))#self.vis_dict[cand]['test_acc'], self.vis_dict[cand]['params']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)

            # if self.epoch == self.max_epochs - 1:
            #     self.eval_test = True

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            # pre_best = max(self.candidates, key=lambda x: self.vis_dict[x]['acc'] * coef_sign)

            self.candidates = mutation + crossover 

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()


    def cand_eval(self, cand_list):
        for cand in cand_list:
            assert isinstance(cand, tuple)
            assert cand in self.vis_dict
            info = self.vis_dict[cand]
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            sampled_config = {}
            sampled_config['layer_num'] = depth
            sampled_config['mlp_ratio'] = mlp_ratio
            sampled_config['num_heads'] = num_heads
            sampled_config['embed_dim'] = [embed_dim] * depth
        
            
            test_stats = evaluate(self.test_loader, self.model, self.device, amp=self.args.amp, mode='retrain', retrain_config=sampled_config)
            info['test_acc'] = test_stats['acc1']
            info['test_acc5'] = test_stats['acc5']

    def eval_topk(self):
        topk_cand_list = []
        for i, cand in enumerate(self.keep_top_k[50]):
            topk_cand_list.append(cand)

        self.cand_eval(topk_cand_list)

        i = 0
        for cand in topk_cand_list:
            info = self.vis_dict[cand]
            i += 1
            print('No.{} test_acc is {}'.format(i, info['test_acc']))
            
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=23)
    parser.add_argument('--min-param-limits', type=float, default=18)

    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # custom model argument
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'EVO_IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    # Kendall tau
    parser.add_argument('--kendall', action='store_true', default=False, help='Calc Kendall tau')
    parser.add_argument('--save_ckpt_path', default='', type=str, help='Save acc top-1 for kendall')
    parser.add_argument('--load_ckpt_path', default='', type=str, help='Load acc top-1 for kendall')
    parser.add_argument('--cluster_json', default='cls_cluster.json', type=str, help='Path to load kendall search cluster')
    parser.add_argument('--kendall_record_json', default='kendall_search.json', type=str, help='Path to save kendall search record')
    parser.add_argument('--kendall_search', action='store_true', default=False, help='Search Kendall tau')

    # proxy
    parser.add_argument('--record_json', default='', type=str, help='Path to load kendall search record')
    parser.add_argument('--measure-name', default='kd_kl_divergence', type=str, help='Measure used to calculate the score of proxy')
    parser.add_argument('--data-load', default='kd', type=str, help='Data load method of minibatch, one of kd, zc_imgs, random, grasp.')

    return parser

def main(args):

    update_config_from_file(args.cfg)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # save config for later experiments
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
        f.write(args_text)
    # fix the seed for reproducibility

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher

    dataset_test, args.nb_classes = build_dataset(is_train=False, args=args, folder_name="val")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(2 * args.batch_size),
        sampler=sampler_test, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    print(f"Creating SuperVisionTransformer")
    print(cfg)
    model = Vision_TransformerSuper(img_size=args.input_size,
                                    patch_size=args.patch_size,
                                    embed_dim=cfg.SUPERNET.EMBED_DIM, depth=cfg.SUPERNET.DEPTH,
                                    num_heads=cfg.SUPERNET.NUM_HEADS,mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                    qkv_bias=True, drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    gp=args.gp,
                                    num_classes=args.nb_classes,
                                    max_relative_position=args.max_relative_position,
                                    relative_position=args.relative_position,
                                    change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.teacher_model:
        teacher_model = create_model(
            args.teacher_model,
            pretrained=True,
            num_classes=args.nb_classes,
        )
        teacher_model.to(device)
        if args.distributed:
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu])
    else:
        teacher_model = None

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.kendall_search:
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            print("resume from checkpoint: {}".format(args.resume))
            model_without_ddp.load_state_dict(checkpoint['model'])

    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}

    size = int((256 / 224) * args.input_size)
    zc_transforms = build_transform(False, args)
    t_data_config = resolve_model_data_config(teacher_model)
    t_transforms = create_transform(**t_data_config, is_training=False)

    searcher = EvolutionSearcher(args, device, model, model_without_ddp, choices, data_loader_test, args.output_dir, os.path.join(args.data_path, 'train'), (zc_transforms, t_transforms), args.kendall, teacher_model)
    
    if args.load_ckpt_path:
        searcher.load_checkpoint(args.load_ckpt_path)
        searcher.eval_test = False
    
    
    begin_time = time.time()
    if args.kendall_search:
        record = searcher.kendall_search(N=20, cluster_json=args.cluster_json)
        with open(args.kendall_record_json, "w") as f:
            json.dump(record, f)
    else:
        searcher.search(args.record_json)
    end_time = time.time()

    print('total wall time = {:.2f} seconds'.format(end_time - begin_time))
    print('total wall time = {:.2f} hours'.format((end_time - begin_time) / 3600))

    if args.kendall_search:
        pass
    else:
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            print("resume from checkpoint: {}".format(args.resume))
            model_without_ddp.load_state_dict(checkpoint['model'])
        searcher.eval_topk()

    if args.save_ckpt_path:
        searcher.save_checkpoint(args.save_ckpt_path)
    else:
        searcher.save_checkpoint()
    

    # trans ckpt to json
    
    def load_checkpoint(checkpoint_path=""):
        if not os.path.exists(checkpoint_path):
            return None
        info = torch.load(checkpoint_path)
        memory = info['memory']
        candidates = info['candidates']
        vis_dict = info['vis_dict']
        keep_top_k = info['keep_top_k']
        return memory, candidates, vis_dict, keep_top_k

    def ckpt_to_json(path_in, path_out):
        memory, candidates, vis_dict, keep_top_k = load_checkpoint(path_in)
        # read memory to json
        res_list = []
        for indivs in memory:
            res = []
            for indiv in indivs:
                cfg = indiv
                info = vis_dict[indiv]
                res.append({"cfg": cfg, "info": info})
            res_list.append(res)
        # read cand to json
        res = []
        for indiv in candidates:
            cfg = indiv
            info = vis_dict[indiv]
            res.append({"cfg": cfg, "info": info})
        res_list.append(res)

        res.clear()
        for i, indiv in enumerate(keep_top_k[50]):
            cfg = indiv
            info = vis_dict[indiv]
            # print(info)
            res.append({"cfg": cfg, "info": info})
        res_dict = {"topk": res, "info": res_list}
        # write to json
        with open(path_out, "w") as f:
            json.dump(res_dict, f)
    
    checkpoint_path = os.path.join(args.output_dir, "checkpoint-{}.pth.tar".format(searcher.epoch))
    json_path = os.path.join(args.output_dir, args.output_dir.split('/')[-1] + "-checkpoint-{}.json".format(searcher.epoch))
    if args.output_dir:
        ckpt_to_json(checkpoint_path, json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAZC - AutoFormer evolution search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
