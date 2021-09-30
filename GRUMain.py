"""
主函数
"""
from Model import *
import pandas as pd
import argparse
# 加载向量
import scipy.io as sio
from utils import *
import metric as error
from collections import defaultdict
from spider.github.githubDataFromRemote.redisClient import redisDB

redis = redisDB.connect()
# 用户已经点击过的项
used_items = defaultdict(set)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=r'D:\work\learning\spider\modal\dataset\embedding9.mat',
                    help='图嵌入向量')
parser.add_argument('--epochs', type=int, default=100, help='训练的次数')
parser.add_argument('--batch-size', type=int, default=512, help='每次输入的样本数量')
parser.add_argument('--hidden-size', type=int, default=128, help='hidden state size')
parser.add_argument('--out-size', type=int, default=128, help='这个要和节点向量保持一致')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 惩罚系数')
parser.add_argument('--lr', type=float, default=0.009, help='学习率')
parser.add_argument('--step', type=int, default=3, help='gnn 单元的层数')
parser.add_argument('--lr-dc', type=float, default=0.96, help='学习率衰减率,学习率=学习率*衰减率')
parser.add_argument('--lr_dc_step', type=int, default=100, help='学习率下降的步骤数，没100就乘以衰减率')
parser.add_argument('--topK', type=int, default=10, help='保留前K个')
path_feat = r'D:\work\learning\spider\modal\dataset\feature.csv'
#其它特征向量
path_session = r'D:\work\learning\spider\modal\dataset\...'#训练集
path_session_test = r'D:\work\learning\spider\modal\dataset\...'#测试集
opt = parser.parse_args()
train_data = sio.loadmat(opt.dataset)
ccc = train_data['embedding']
feat = concat(path_feat)
node_sum, dimension = ccc.shape

model = GGNN(dimension=dimension, f_dim=4, f_out_size=20, item_n=3126, hidden_size=64,
             out_size=64,
             batch_size=opt.batch_size,
             n_node=node_sum, lr=0.0099, l2=0.1, decay=opt.lr_dc_step, step=opt.step, lr_dc=opt.lr_dc,
             alpha=0.9, beta=0.001)


def train():

    best_result = {}
    best_epoch = {}
    print("point 1")
    for k in [5, 10, 15, 20]:
        best_result[k] = [0, 0, 0]
        best_epoch[k] = [0, 0, 0]
    for epoch in range(opt.epochs):
        total_loss = []
        print("-----------------------{}".format(epoch))
        # 输出item
        # 这个用户，排除已经存在session中的item
        trained_items = {}
        fetch = [model.vars, model.opt, model.loss_train, model.logits, model.global_step, model.learning_rate]
        target = []
        logits = []
        for items, tar, user, n_session in generate_batch(path_session, model.batch_size, shuffle=False):

            u, _, loss, logit, step, learn = model.run(fetch, tar=tar, item=items, embeddings=ccc,
                                                       n_session=n_session, feat=feat)
            for u, ites, ta in zip(user, items, tar):
                used_items[u].update(ites + [ta])
            total_loss.append(loss)
            target.extend(tar)
            logits.extend(np.argmax(logit, axis=1))
        test_loss_ = []
        hit, mrr, NDCG = {}, {}, {}
        for k in [5, 10, 15, 20]:
            hit[k] = []
            mrr[k] = []
            NDCG[k] = []
        for items, tar, user, n_session in generate_batch(path_session_test, model.batch_size,  shuffle=True):
            score, test_loss = model.run([model.test_logits, model.loss_test], tar=tar, item=items,
                                         embeddings=ccc,
                                         n_session=n_session, feat=feat)
            for u, ites in zip(user, items):
                used_items[u].update(ites)
            test_loss_.append(test_loss)
            for u, sco in zip(user, score):
                itd = list(used_items[u])
                sco[itd] = -np.inf
            for k in [5, 10, 15, 20]:
                index = np.argsort(-score, 1)[:, :k]
                n, h, m = model_score(index, tar, k)
                hit[k] = hit[k] + h
                mrr[k] = mrr[k] + m
                NDCG[k] = NDCG[k] + n

        for k in [5, 10, 15, 20]:
            print("k={};HR@k:{},MRR@K:{},NDCG@K:{}".format(k, np.mean(hit[k]), np.mean(mrr[k]),
                                                           np.mean(NDCG[k])))
            acc_show(hit[k], mrr[k], NDCG[k], best_result[k], best_epoch[k], test_loss_, epoch, k)


def model_score(index, tar, k):
    NDCG, hit, mrr = [], [], []
    for score, target in zip(index, tar):
        IDCG = 0
        # 理想情况下
        IDCG += 1 / math.log(2, 2)
        DCG = 0
        l = list(score)
        NDCG.append(error.ndcg(l, [target]))
        hit.append(error.hit(l, [target]))
        mrr.append(error.mrr(l, [target]))

    return NDCG, hit, mrr


def acc_show(hit, mrr, NDCG, best_result, best_epoch, test_loss_, epoch, k):
    hit = np.mean(hit, axis=0)
    mrr = np.mean(mrr, axis=0)
    NDCG = np.mean(NDCG, axis=0)
    test_loss = np.mean(test_loss_)
    print(test_loss)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = epoch
    if NDCG >= best_result[2]:
        best_result[2] = NDCG
        best_epoch[2] = epoch
    print("k={};HR@k:{},MRR@K:{},NDCG@K:{}".format(k, best_result[0], best_result[1], best_result[2]))
    print("取得最大精度的训练次数：{}".format(best_epoch))


if __name__ == '__main__':
    train()
