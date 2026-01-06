import torch
from model.CME import ACME, ACMELayer
from model.GAT import GATLayer
from model.MultiHeadAttention import MultiHeadAttention
from model.PairNN import PairNN
import torch.nn as nn
from transformers import BertModel
from model.Transformer import AdditiveAttention, TransEncoder
from model.UME import RUME, RUMELayer
from model.baselines.rankcp.rank_cp import RankNN
# from pretrained_models.textual.RoBERTa.SRoBERTa import RobertaModel, RobertaPreTrainedModel
from transformers import Wav2Vec2Model, RobertaModel, RobertaPreTrainedModel
from utils.global_variables import DEVICE, EMOTION_MAPPING, GRAPH_CONFIG_T, HYPERGRAPH_CONFIG_T, HYPERGRAPH_CONFIG_T_A, HYPERGRAPH_CONFIG_T_A_V
import torch.nn.functional as F
import dgl
import json





def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


# class MECPECModel(nn.Module):
#     _keys_to_ignore_on_load_missing = [r"position_ids"]
#     def __init__(self, lm_config, config, data_name="ECF", activation='relu', modality = ['textual']):
class MECPECModel(nn.Module):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, lm_config, config, data_name="ECF", activation='relu', modality=['textual']):
        super(MECPECModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.textual_pretrain_model_dir)
        # super(MECPECModel, self).__init__()
        self.gcn_dim = config.hidden_dim
        GRAPH = HYPERGRAPH_CONFIG_T
        if data_name == "ECF":
            prior_path = "data/ECF/prior_knowledge.json"
        if data_name == "MECAD":
            prior_path = "data/MECAD/prior_knowledge1.json"
        if 'textual' in modality:
            # self.bert = BertModel(lm_config)
            self.attention_head_size = int(lm_config.hidden_size / lm_config.num_attention_heads)
            self.turnAttention = MultiHeadAttention(lm_config.num_attention_heads, lm_config.hidden_size,
                                                    self.attention_head_size, self.attention_head_size,
                                                    lm_config.attention_probs_dropout_prob)
            self.linear_t = nn.Linear(lm_config.hidden_size, self.gcn_dim)

        if 'audio' in modality:
            self.linear_a = nn.Linear(6373, self.gcn_dim)
            rumeLayer = RUMELayer(feature_size=self.gcn_dim, dropout=config.rnn_drop, rnn_type=config.rnn_type,
                                  use_vanilla=config.use_vanilla, use_rnnpack=config.use_rnnpack, no_cuda=False)
            self.rume = RUME(rumeLayer, num_layers=config.rnn_n_layers)
            GRAPH = HYPERGRAPH_CONFIG_T_A
        if 'video' in modality:
            self.linear_v = nn.Linear(4096, self.gcn_dim)
            rumeLayer = RUMELayer(feature_size=self.gcn_dim, dropout=config.rnn_drop, rnn_type=config.rnn_type,
                                  use_vanilla=config.use_vanilla, use_rnnpack=config.use_rnnpack, no_cuda=False)
            self.rume = RUME(rumeLayer, num_layers=config.rnn_n_layers)
            GRAPH = HYPERGRAPH_CONFIG_T_A_V

            # self.attention_head_size = int(768 / lm_config.num_attention_heads)
            # self.turnAttention = MultiHeadAttention(lm_config.num_attention_heads, 768,
            #                                     self.attention_head_size, self.attention_head_size,
            #                                     lm_config.attention_probs_dropout_prob)

        acmeLayer = ACMELayer(feature_size=self.gcn_dim, nheads=config.cross_num_head, dropout=config.cross_drop,
                              no_cuda=config.no_cuda)
        self.acme = ACME(acmeLayer, num_layers=config.cross_n_layers)
        self.linear_cat = nn.Linear(2 * self.gcn_dim, self.gcn_dim)
        self.drop_cat = nn.Dropout(config.cross_drop)
        self.transform = nn.Linear(self.gcn_dim, self.gcn_dim)

        #         # self.bert = BertModel.from_pretrained(config.pretrain_model_dir)
        #         self.xl_net =XLNetModel.from_pretrained(config.pretrain_model_dir)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."
        self.dropout = nn.Dropout(config.dropout)

        # self.emotion_mlp = EmotionMLP(2 * self.gcn_dim, len(set(EMOTION_MAPPING[data_name].values())))
        # self.cause_mlp = CauseMLP(2 * self.gcn_dim)

        self.emotion_mlp = EmotionMLP(2 * self.gcn_dim, len(set(EMOTION_MAPPING[data_name].values())))
        self.emotion_mlp_cf = EmotionMLP(2 * self.gcn_dim, len(set(EMOTION_MAPPING[data_name].values())))
        self.cause_mlp = CauseMLP(2 * self.gcn_dim)

        # 转化为图神经网络的维度

        self.gat_layers = config.gat_layers
        self.graph_attention_size = int(self.gcn_dim / config.num_graph_attention_heads)
        # 图注意力层
        self.GAT_layers = nn.ModuleList([GATLayer(meta_paths=GRAPH['hyper_meta_paths'],
                                                  in_size=self.gcn_dim, out_size=self.graph_attention_size,
                                                  layer_num_heads=config.num_graph_attention_heads) for _ in
                                         range(self.gat_layers)])
        self.ffn_layers = nn.ModuleList(
            [PositionWiseFeedForward(self.gcn_dim, self.gcn_dim, 0.2) for _ in range(self.gat_layers)])
        # self.gru_conversation = nn.GRU(self.gcn_dim, self.gcn_dim, batch_first=False)
        # self.gru_utterance = nn.GRU(self.gcn_dim, self.gcn_dim, batch_first=False)
        # self.hgc_layers = nn.ModuleList([HeteroGraphConvLayer(in_size=self.gcn_dim, out_size=self.graph_attention_size,
        #                                           layer_num_heads=config.num_graph_attention_heads) for _ in range(self.gat_layers)])
        # 情绪原因配对
        self.ec_pairing = PairNN(2 * self.gcn_dim, config.pos_emb_dim, config.rel_pos_k)

        self.mask_logits = nn.ModuleDict({
            'utterance_t': nn.Linear(self.gcn_dim, 1),
        })
        self.modal_weights = nn.Parameter(torch.tensor([1.0]))

        self.use_llm = config.use_llm

        # emotion-conditioned causal mask scorer
        # self.mask_logits_emotion = nn.Linear((len(modality) + 1) * self.gcn_dim, 1)

        self.mask_logits_emotion = nn.Sequential(
            nn.Linear(2 * self.gcn_dim, self.gcn_dim),
            nn.ReLU(),
            nn.Linear(self.gcn_dim, 1)
        )

        with open(prior_path, "r", encoding="utf-8") as f:
            self.prior_knowledge = json.load(f)

    def forward(self,
                conversation_ids=None,
                input_ids=None,
                attention_masks=None,
                token_type_ids=None,
                position_ids=None,
                head_masks=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False,
                speaker_ids=None,
                mention_ids=None,
                emotion_ids=None,
                turn_masks=None,
                uttr_indices=None,
                graphs=None,
                uttr_len=None,
                audio_features=None,
                video_features=None,
                modality=[],
                is_training=True):
        em_logits, ec_logits = 0, 0
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.size(0)

        # 清理显存的函数
        def clear_cache(*vars):
            for var in vars:
                del var
            torch.cuda.empty_cache()

        if 'textual' in modality:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_masks,
                token_type_ids=None,
                position_ids=position_ids,
                # speaker_ids=speaker_ids,
                head_mask=head_masks,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # outputs = self.bert(input_ids=input_ids,
            #                 attention_mask=attention_masks,
            #                 token_type_ids=token_type_ids)
            # outputs = self.xl_net(input_ids=input_ids,
            #                         attention_mask=attention_masks,
            #                         token_type_ids=token_type_ids)
            # B: batch size
            # N: 对话的话语长度（根据batch进行padding）
            # S：对话输入的序列token长度，填充至max-seq-length
            # D: Roberta的隐藏层维度
            # M: 这个batch的所有图的节点数之和
            # H: 图神经网络的隐层维度

            sequence_outputs_h = outputs[0]  # B*S*D
            pooled_outputs = outputs[1]  # B*D
            sequence_outputs_h, _ = self.turnAttention(sequence_outputs_h, sequence_outputs_h, sequence_outputs_h,
                                                       turn_masks)  # turn_masks: B*S*S
            sequence_outputs_h = self._batched_index_select(sequence_outputs_h, uttr_indices, mention_ids)  # B*N*D
            sequence_outputs_h = self.linear_t(sequence_outputs_h.to(self.linear_t.weight.device))
            pooled_outputs = self.linear_t(pooled_outputs).unsqueeze(1)
            # 清理不再使用的变量
            clear_cache(outputs)
            # pooled_outputs = torch.mean(sequence_outputs, dim=1)  # B*D
        if 'audio' in modality:
            conv_audio_in = torch.mean(audio_features, dim=1).unsqueeze(1)
            audio_in = self.linear_a(audio_features)
            conv_audio_in = self.linear_a(conv_audio_in)
            audio_in = self.rume(audio_in, uttr_len)
            conv_audio_in = self.rume(conv_audio_in, uttr_len)
            # 清理不再使用的变量
            clear_cache(audio_features)
        if 'video' in modality:
            conv_video_in = torch.mean(video_features, dim=1).unsqueeze(1)
            video_in = self.linear_v(video_features)
            conv_video_in = self.linear_v(conv_video_in)
            video_in = self.rume(video_in, uttr_len)
            conv_video_in = self.rume(conv_video_in, uttr_len)
            # 清理不再使用的变量
            clear_cache(video_features)

        # # ACME 跨模态注意力
        # audio_mask = (uttr_indices != 0).float()
        # featcross_t, featcross_a = self.acme(sequence_outputs_h, audio_in, audio_mask)  # audio_mask: B * S
        # conv_featcross_t, conv_featcross_a = self.acme(pooled_outputs, conv_audio_in, audio_mask[:, 0].unsqueeze(-1))
        #
        # featcross_cat = torch.concat([featcross_t, featcross_a], dim=-1)
        # sequence_outputs_h = self.drop_cat(F.relu(self.linear_cat(featcross_cat)))
        # conv_featcross_cat = torch.concat([conv_featcross_t, conv_featcross_a], dim=-1)
        # pooled_outputs = self.drop_cat(F.relu(self.linear_cat(conv_featcross_cat)))
        # # 清理不再使用的变量
        # clear_cache(featcross_t, featcross_a, conv_featcross_t, conv_featcross_a, featcross_cat, conv_featcross_cat)
        # graph_in = self.transform(sequence_outputs_h)  # B*N*H

        # initialize graph nodes
        h_dict = {'utterance_t': None, 'emotion': None, 'cause': None, 'hyperedge': None}  # 用于存储不同类型的节点特征
        if 'audio' in modality:
            h_dict = {
                'utterance_t': None, 'utterance_a': None,
                'emotion': None, 'cause': None, 'hyperedge': None
            }
        if 'video' in modality:
            h_dict = {
                'utterance_t': None, 'utterance_a': None, 'utterance_v': None,
                'emotion': None, 'cause': None, 'hyperedge': None
            }
        # h_dict = {
        #     'utterance_t': None, 'utterance_a': None, 'utterance_v': None,
        #     'emotion': None, 'cause': None, 'hyperedge': None
        # }
        # h_dict = {'utterance_t':None,'utterance_a':None, 'utterance_v':None,
        #           'conversation_t':None,'conversation_a':None,'conversation_v':None,
        #           'emotion':None,'cause':None} # 用于存储不同类型的节点特征
        # 看是否需要注释
        graph_in = sequence_outputs_h # B*N*H
        # initialize graph nodes


        # for i in range(len(graphs)):
        #     N = uttr_len[i]
        #
        #     # 切分文本多模态特征
        #     feat_t = sequence_outputs_h[i][:N]  # (N, H)
        #     feat_a = audio_in[i][:N] if 'audio' in modality else None  # (N, H)
        #     feat_v = video_in[i][:N] if 'video' in modality else None  # (N, H)
        #
        #     global_h_i = pooled_outputs[i]  # (1, H)
        #
        #     # === 存入 utterance 多模态（注意按类型堆叠）===
        #     for key, feat in {
        #         'utterance_t': feat_t,
        #         'utterance_a': feat_a,
        #         'utterance_v': feat_v
        #     }.items():
        #         if feat is None:
        #             continue
        #         h_dict[key] = torch.cat([h_dict[key], feat], dim=0) if h_dict[key] is not None else feat
        #
        #     # === emotion & cause 模态继承文本特征 ===
        #     for key in ['emotion', 'cause']:
        #         h_dict[key] = torch.cat([h_dict[key], feat_t], dim=0) if h_dict[key] is not None else feat_t
        #
        #     # === hyperedge = [multi-modal stacked [N nodes] + global pooled] ===
        #     global_modalities = [global_h_i]  # 默认文本
        #     if 'audio' in modality:
        #         global_modalities.append(global_h_i)  # 可以考虑用 conv_audio_in 的 pooled
        #     if 'video' in modality:
        #         global_modalities.append(global_h_i)  # 可以考虑用 conv_video_in 的 pooled
        #     global_modalities = torch.cat(global_modalities, dim=0)  # (3, H)
        #     hyper_f = torch.cat([feat_t, global_modalities], dim=0)  # (N+3, H)
        #     h_dict['hyperedge'] = torch.cat([h_dict['hyperedge'], hyper_f], dim=0) if h_dict[
        #                                                                                   'hyperedge'] is not None else hyper_f

        for i in range(len(graphs)):
            N = uttr_len[i]

            # 切分文本多模态特征
            feat_t = sequence_outputs_h[i][:N]  # (N, H) (N,512)
            feat_a = audio_in[i][:N] if 'audio' in modality else None  # (N, H) (N, 512)
            feat_v = video_in[i][:N] if 'video' in modality else None  # (N, H)

            # global_h_i = pooled_outputs[i]  # (1, H) (1, 512)

            # === 存入 utterance 多模态（注意按类型堆叠）===
            for key, feat in {
                'utterance_t': feat_t,
                'utterance_a': feat_a,
                'utterance_v': feat_v
            }.items():
                if feat is None:
                    continue
                h_dict[key] = torch.cat([h_dict[key], feat], dim=0) if h_dict[key] is not None else feat

            # === emotion & cause 模态继承文本特征 ===要改
            for key in ['emotion', 'cause']:
                feats = [feat_t]  # 文本一定存在
                if 'audio' in modality and feat_a is not None:
                    feats.append(feat_a)
                if 'video' in modality and feat_v is not None:
                    feats.append(feat_v)

                # 多模态平均
                multimodal_feat = torch.stack(feats, dim=0).mean(dim=0)  # (N, H)

                h_dict[key] = torch.cat([h_dict[key], multimodal_feat], dim=0) if h_dict[
                                                                                      key] is not None else multimodal_feat

            # === hyperedge = [multi-modal stacked [N nodes] + prior + global pooled] ===
            G = graphs[i]
            num_hyper = G.num_nodes('hyperedge')
            num_global = len(modality)
            feat_dim = feat_t.size(-1)
            uttr_types = []
            if 'textual' in modality:
                uttr_types.append('utterance_t')
            if 'audio' in modality:
                uttr_types.append('utterance_a')
            if 'video' in modality:
                uttr_types.append('utterance_v')
            hyperedge_h_i = torch.zeros(num_hyper, feat_dim, device=feat_t.device)
            # 模态间超边
            for u in range(N):
                feats = []
                if 'textual' in modality:
                    feats.append(feat_t[u])
                if 'audio' in modality:
                    feats.append(feat_a[u])
                if 'video' in modality:
                    feats.append(feat_v[u])

                # 多模态平均 / concat 后线性都行，这里用 mean
                hyperedge_h_i[u] = torch.stack(feats, dim=0).mean(dim=0)
            if self.use_llm:
                # 先验超边（candidate / latent）
                for hid in range(N, num_hyper - num_global):
                    # 默认权重为 None
                    w = None

                    if G.in_edges(hid, etype=('emotion', 'cand_e', 'hyperedge'))[0].numel() > 0:
                        w = 1.0
                        emo_idx = G.predecessors(hid, etype=('emotion', 'cand_e', 'hyperedge'))[0]

                        utt_feats = []
                        for uttr_type in uttr_types:
                            uids = G.predecessors(
                                hid, etype=(uttr_type, f'cand_link_{uttr_type[-1]}', 'hyperedge')
                            )
                            if len(uids) == 0:
                                continue

                            if uttr_type == 'utterance_t':
                                utt_feats.append(feat_t[uids])
                            elif uttr_type == 'utterance_a':
                                utt_feats.append(feat_a[uids])
                            elif uttr_type == 'utterance_v':
                                utt_feats.append(feat_v[uids])

                    # ---------- latent ----------
                    elif G.in_edges(hid, etype=('emotion', 'latent_e', 'hyperedge'))[0].numel() > 0:
                        w = 0.3
                        emo_idx = G.predecessors(hid, etype=('emotion', 'latent_e', 'hyperedge'))[0]

                        utt_feats = []
                        for uttr_type in uttr_types:
                            uids = G.predecessors(
                                hid, etype=(uttr_type, f'latent_link_{uttr_type[-1]}', 'hyperedge')
                            )
                            if len(uids) == 0:
                                continue

                            if uttr_type == 'utterance_t':
                                utt_feats.append(feat_t[uids])
                            elif uttr_type == 'utterance_a':
                                utt_feats.append(feat_a[uids])
                            elif uttr_type == 'utterance_v':
                                utt_feats.append(feat_v[uids])

                    # 如果不是 candidate/latent 超边，跳过
                    if w is None or len(uids) == 0:
                        continue

                    # 计算 emotion 特征
                    emo_feat = h_dict['emotion'][emo_idx]
                    # 计算 utterance 平均特征
                    utt_feat = torch.cat(utt_feats, dim=0).mean(dim=0)
                    # 融合并乘上权重
                    hyperedge_h_i[hid] = 0.5 * emo_feat + 0.5 * utt_feat

            # 模态内超边
            g_idx = 0
            if 'textual' in modality:
                hyperedge_h_i[num_hyper - num_global + g_idx] = pooled_outputs[i].squeeze(0)
                g_idx += 1

            if 'audio' in modality:
                hyperedge_h_i[num_hyper - num_global + g_idx] = conv_audio_in[i].squeeze(0)
                g_idx += 1

            if 'video' in modality:
                hyperedge_h_i[num_hyper - num_global + g_idx] = conv_video_in[i].squeeze(0)

            if h_dict['hyperedge'] is not None:
                h_dict['hyperedge'] = torch.cat([h_dict['hyperedge'], hyperedge_h_i], dim=0)
            else:
                h_dict['hyperedge'] = hyperedge_h_i


        # construct big graph:
        graph_big = dgl.batch(graphs)

        model_device = next(self.parameters()).device
        graph_big = graph_big.to('cpu').to(model_device)  # 先到CPU再到目标设备
        h_dict = {k: v.to(model_device) for k, v in h_dict.items()}

        uttr_nodes_num = int((graph_big.num_nodes() - batch_size) / 3)  # 除去对话节点的话语节点数/3，因为总共有三个类型节点，除去对话节点

        for layer_num, GAT_layer in enumerate(self.GAT_layers):
            # graph_features : M * H
            graph_features = GAT_layer(graph_big, h_dict)  # M * H
            graph_features = self.ffn_layers[layer_num](graph_features.unsqueeze(1)).squeeze(1)  # M * H

            # 动态按 node type 切片（安全且通用）
            node_counts = {ntype: graph_big.num_nodes(ntype) for ntype in graph_big.ntypes}
            offsets = {}
            offset = 0
            for ntype in graph_big.ntypes:
                cnt = node_counts[ntype]
                offsets[ntype] = (offset, offset + cnt)
                offset += cnt

            # 将切片结果回写到 h_dict（仅写入存在的 key）
            # 注意：graph_big.ntypes 的顺序即为 graph_features 的 node 顺序
            for ntype in graph_big.ntypes:
                start, end = offsets[ntype]
                if end > start:  # 有节点
                    feat = graph_features[start:end]
                else:
                    feat = torch.empty((0, graph_features.size(1)), device=graph_features.device)
                # 只更新我们关心的类型，忽略其他可能存在的类型
                if ntype in h_dict:
                    h_dict[ntype] = feat

        # ================= Emotion-conditioned Counterfactual =================
        # h_dict 已包含:
        #   utterance_t: (sum_N, H)
        #   emotion:     (sum_N, H)

        sparsity_loss = 0.0
        cf_loss = 0.0

        # ---------- 1. 构造 emotion-conditioned 表示 ----------
        # 后面会对齐成 B * N * H
        fea_idx = 0
        max_uttr_num = max(uttr_len)

        mask_batch = []
        h_dict_cf = {k: v.clone() for k, v in h_dict.items()}

        for i in range(len(graphs)):
            conv_id = conversation_ids[i]
            prior_conv = self.prior_knowledge[conv_id]

            N = uttr_len[i]

            # ===== emotion & utterance (per modality) =====
            emo_i = h_dict['emotion'][fea_idx:fea_idx + N]  # (N, H)
            utt_text_i = h_dict['utterance_t'][fea_idx:fea_idx + N]

            utt_feats = []
            if 'textual' in modality:
                utt_feats.append(h_dict['utterance_t'][fea_idx:fea_idx + N])
            if 'audio' in modality:
                utt_feats.append(h_dict['utterance_a'][fea_idx:fea_idx + N])
            if 'video' in modality:
                utt_feats.append(h_dict['utterance_v'][fea_idx:fea_idx + N])

            # 融合多模态 utterance（用于 mask 预测）
            # 方式 1：concat（最稳妥）
            utt_i = torch.cat(utt_feats, dim=-1)  # (N, H*num_modality)

            # ---------- 2. emotion-conditioned mask ----------
            cond_feat = torch.cat([emo_i, utt_text_i], dim=-1)
            logits = self.mask_logits_emotion(cond_feat).squeeze(-1)  # (N,)
            base_prob = logits
            if self.use_llm:
                # ===== prior bias（不变）=====
                prior_bias = torch.zeros(N, device=logits.device)
                for t in range(N):
                    t_key = str(t + 1)
                    if t_key in prior_conv and len(prior_conv[t_key]) > 0:
                        max_prior = max(prior_conv[t_key].values())
                    else:
                        max_prior = 0.0

                    # prior_bias[t] = max_prior - 0.5
                    prior_bias[t] = max_prior

                base_prob = torch.sigmoid(logits + 0.5 * prior_bias) # 这边是先验权重

            mm_gate = 1.0
            mm_count = 0

            if 'audio' in modality:
                a_i = h_dict['utterance_a'][fea_idx:fea_idx + N]
                mm_gate = mm_gate * torch.sigmoid(a_i.norm(dim=-1))
                mm_count += 1

            if 'video' in modality:
                v_i = h_dict['utterance_v'][fea_idx:fea_idx + N]
                mm_gate = mm_gate * torch.sigmoid(v_i.norm(dim=-1))
                mm_count += 1

            if mm_count > 0:
                mm_gate = mm_gate ** (1 / mm_count)  # geometric mean

            probs = base_prob * mm_gate

            # STE
            hard = (probs > 0.5).float()
            mask_i = (hard - probs).detach() + probs  # (N,)

            # ---------- 3. 反事实：mask = 1 屏蔽该 utterance（所有模态） ----------
            for uttr_type in ['utterance_t', 'utterance_a', 'utterance_v']:
                if uttr_type in h_dict_cf:
                    utt_feat_i = h_dict[uttr_type][fea_idx:fea_idx + N]
                    # 高斯噪声（与特征尺度匹配）
                    # noise = torch.randn_like(utt_feat_i) * utt_feat_i.std(dim=0, keepdim=True).detach()
                    # utt_cf_i = utt_feat_i + 0.1 * noise * mask_i.unsqueeze(-1)
                    beta = 0.7
                    perm = torch.randperm(utt_feat_i.size(0), device=utt_feat_i.device)
                    noise = utt_feat_i[perm]

                    utt_cf_i = utt_feat_i * (1 - mask_i.unsqueeze(-1)) \
                               + (beta * utt_feat_i + (1 - beta) * noise) * mask_i.unsqueeze(-1)

                    # utt_cf_i = (utt_feat_i * (1 - mask_i).unsqueeze(-1) + noise * mask_i.unsqueeze(-1))
                    # utt_cf_i = utt_feat_i * (1 - mask_i).unsqueeze(-1)
                    h_dict_cf[uttr_type][fea_idx:fea_idx + N] = utt_cf_i

            # ---------- 4. padding 到 batch ----------
            padded_mask_i = F.pad(mask_i, (0, max_uttr_num - N))
            mask_batch.append(padded_mask_i.unsqueeze(0))

            # ---------- 5. sparsity（鼓励少量证据） ----------
            sparsity_loss += mask_i.mean()

            fea_idx += N

        mask = torch.cat(mask_batch, dim=0)  # (B, max_uttr_num)

        sparsity_loss = sparsity_loss / len(graphs)


        for layer_num, GAT_layer in enumerate(self.GAT_layers):
            # graph_features : M * H
            graph_features = GAT_layer(graph_big, h_dict_cf)  # M * H
            graph_features = self.ffn_layers[layer_num](graph_features.unsqueeze(1)).squeeze(1)  # M * H

            # 动态按 node type 切片（安全且通用）
            node_counts = {ntype: graph_big.num_nodes(ntype) for ntype in graph_big.ntypes}
            offsets = {}
            offset = 0
            for ntype in graph_big.ntypes:
                cnt = node_counts[ntype]
                offsets[ntype] = (offset, offset + cnt)
                offset += cnt

            # 将切片结果回写到 h_dict（仅写入存在的 key）
            # 注意：graph_big.ntypes 的顺序即为 graph_features 的 node 顺序
            for ntype in graph_big.ntypes:
                start, end = offsets[ntype]
                if end > start:  # 有节点
                    feat = graph_features[start:end]
                else:
                    feat = torch.empty((0, graph_features.size(1)), device=graph_features.device)
                # 只更新我们关心的类型，忽略其他可能存在的类型
                if ntype in h_dict_cf:
                    h_dict_cf[ntype] = feat
        graphs = dgl.unbatch(graph_big)

        if h_dict['utterance_t'].dim() > 2:
            h_dict['utterance_t'], h_dict['emotion'], h_dict['cause'], h_dict['hyperedge'] = \
                h_dict['utterance_t'].squeeze(0), h_dict['emotion'].squeeze(0), h_dict['cause'].squeeze(0), h_dict['hyperedge'].squeeze(0)
        # get the output of the last GAT layer
        fea_idx = 0
        max_uttr_num = max(uttr_len)
        gragh_h_emotion = None  # 用于存储图结构的情绪节点特征
        gragh_h_cause = None  # 用于存储图结构的原因节点特征
        # mask = None

        # delta_H = {}
        # if h_dict_cf is not None:
        #     for ntype in h_dict.keys():
        #         delta_H[ntype] = h_dict[ntype] - h_dict_cf[ntype]
        # else:
        #     delta_H = h_dict

        for i in range(len(graphs)):
            node_num = uttr_len[i]  # M， 这里包括了一个对话的对话节点和话语节点
            graph_h_emotion_i = h_dict['emotion'][fea_idx:fea_idx + node_num]
            graph_h_emotion_i_cf = h_dict_cf['emotion'][fea_idx:fea_idx + node_num]
            padded_graph_h_emotion_i = F.pad(graph_h_emotion_i,
                                             (0, 0, 0, max_uttr_num - graph_h_emotion_i.size(0)))  # padding N*H
            padded_graph_h_emotion_i_cf = F.pad(graph_h_emotion_i_cf,
                                             (0, 0, 0, max_uttr_num - graph_h_emotion_i_cf.size(0)))  # padding N*H
            graph_h_cause_i = h_dict['cause'][fea_idx:fea_idx + node_num]
            graph_h_cause_i_cf = h_dict_cf['cause'][fea_idx:fea_idx + node_num]
            padded_graph_h_cause_i = F.pad(graph_h_cause_i,
                                           (0, 0, 0, max_uttr_num - graph_h_cause_i.size(0)))  # padding N*H
            padded_graph_h_cause_i_cf = F.pad(graph_h_cause_i_cf,
                                           (0, 0, 0, max_uttr_num - graph_h_cause_i_cf.size(0)))  # padding N*H
            # mask_i = M[fea_idx:fea_idx + node_num]
            # padded_mask_i = F.pad(mask_i, (0, max_uttr_num - mask_i.size(0))).unsqueeze(0)
            fea_idx += node_num
            if gragh_h_emotion is None:
                gragh_h_emotion = padded_graph_h_emotion_i.unsqueeze(0)
                gragh_h_emotion_cf = padded_graph_h_emotion_i_cf.unsqueeze(0)
            else:
                gragh_h_emotion = torch.cat([gragh_h_emotion, padded_graph_h_emotion_i.unsqueeze(0)], dim=0)
                gragh_h_emotion_cf = torch.cat([gragh_h_emotion_cf, padded_graph_h_emotion_i_cf.unsqueeze(0)], dim=0)
            if gragh_h_cause is None:
                gragh_h_cause = padded_graph_h_cause_i.unsqueeze(0)
                gragh_h_cause_cf = padded_graph_h_cause_i_cf.unsqueeze(0)
            else:
                gragh_h_cause = torch.cat([gragh_h_cause, padded_graph_h_cause_i.unsqueeze(0)], dim=0)
                gragh_h_cause_cf = torch.cat([gragh_h_cause_cf, padded_graph_h_cause_i_cf.unsqueeze(0)], dim=0)
            # if mask is None:
            #     mask = padded_mask_i
            # else:
            #     mask = torch.cat([mask, padded_mask_i], dim=0)


        # 筛选配对
        # em_logits = self.emotion_mlp(graph_in)
        # ca_logits = self.cause_mlp(graph_in)
        # couples_pred, emo_caus_pos = self.ec_pairing(graph_in, graph_in) # B*(N*N) 情绪原因配对logits
        em = torch.cat([graph_in, gragh_h_emotion], dim=-1)
        em_cf = torch.cat([graph_in, gragh_h_emotion_cf], dim=-1)
        em_logits = self.emotion_mlp(torch.cat([graph_in, gragh_h_emotion], dim=-1))  # B*N*E 情绪类别logits
        em_logits_cf = self.emotion_mlp_cf(torch.cat([graph_in, gragh_h_emotion_cf], dim=-1))

        ca_logits = self.cause_mlp(torch.cat([graph_in, gragh_h_cause], dim=-1))  # B*N*1 原因类别logits
        ca_logits_cf = self.cause_mlp(torch.cat([graph_in, gragh_h_cause_cf], dim=-1))  # B*N*1 原因类别logits

        emotion_g_h = torch.cat([graph_in, gragh_h_emotion], dim=-1)
        cause_g_h = torch.cat([graph_in, gragh_h_cause], dim=-1)
        emotion_g_h_cf = torch.cat([graph_in, gragh_h_emotion_cf], dim=-1)
        cause_g_h_cf = torch.cat([graph_in, gragh_h_cause_cf], dim=-1)
        couples_pred, emo_caus_pos = self.ec_pairing(emotion_g_h, cause_g_h)  # B*(N*N) 情绪原因配对logits
        couples_pred_cf, _ = self.ec_pairing(emotion_g_h_cf, cause_g_h_cf)

        # em_logits = self.emotion_mlp(gragh_h) # B*N*E 情绪类别logits

        # em_logits = em_logits - em_logits_cf
        # ca_logits = ca_logits - ca_logits_cf
        # couples_pred = couples_pred - couples_pred_cf

        return em_logits, em_logits_cf, ca_logits, ca_logits_cf, couples_pred, couples_pred_cf, emo_caus_pos, em, em_cf, sparsity_loss, mask

    def _audio_batched_index_select(self, audio_features, audio_uttr_indices):
        max_uttr_len = audio_uttr_indices.size(1)
        feature_dim = audio_features.size(-1)
        batch_size = audio_features.size(0)
        doc_sents_h = get_cuda(torch.zeros(batch_size, max_uttr_len, feature_dim))
        for i in range(audio_features.size(0)):
            uttr_idx = 1
            while uttr_idx < max_uttr_len and audio_uttr_indices[i, uttr_idx] != 0:
                doc_sents_h[i, uttr_idx - 1, :] = torch.mean(
                    audio_features[i, audio_uttr_indices[i][uttr_idx - 1]: audio_uttr_indices[i][uttr_idx], :], dim=0)
                uttr_idx += 1
            if audio_uttr_indices[i, uttr_idx - 1] != 0:
                doc_sents_h[i, uttr_idx - 1, :] = torch.mean(audio_features[i, audio_uttr_indices[i][uttr_idx - 1]:, :],
                                                             dim=0)
        return doc_sents_h

    def _batched_index_select(self, sequence_outputs, uttr_indices, mention_ids):
        max_uttr_len = uttr_indices.size(1)
        slen = sequence_outputs.shape[1]
        feature_dim = sequence_outputs.size(-1)
        batch_size = sequence_outputs.size(0)
        # 初始化一个填充张量
        doc_sents_h = get_cuda(torch.zeros(batch_size, max_uttr_len, feature_dim))
        for i in range(sequence_outputs.size(0)):  # 对于每个batch
            device = sequence_outputs.device  # 模型当前 GPU

            sequence_outputs = sequence_outputs.to(device)
            uttr_indices = uttr_indices.to(device)
            mention_ids = mention_ids.to(device)

            sequence_output = sequence_outputs[i]
            mention_num = torch.max(mention_ids[i])  # 话语数
            mention_index = (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen)  # N * S (4*138)
            mentions = mention_ids[i].unsqueeze(0).expand(mention_num, -1)  # shape:N*512
            select_metrix = (mention_index == mentions.to(mention_index.device)).float()
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # 每个话语的词数
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
            uttr_h = torch.mm(select_metrix.to(device), sequence_output.to(device))  # 根据话语长度做加权平均
            doc_sents_h[i, :uttr_h.size(0), :] = uttr_h
        # dummy = uttr_indices.unsqueeze(2).expand(uttr_indices.size(0), uttr_indices.size(1), sequence_outputs.size(2))
        # doc_sents_h = sequence_outputs.gather(1, dummy)
        return doc_sents_h

    # def _batched_index_select(self, sequence_outputs , uttr_indices):
    #     dummy = uttr_indices.unsqueeze(2).expand(uttr_indices.size(0), uttr_indices.size(1), sequence_outputs.size(2))
    #     doc_sents_h = sequence_outputs.gather(1, dummy)
    #     return doc_sents_h

    def loss_pre_emo(self, pred_e, gold_e):
        uttr_len = pred_e.size(1)
        gold_e = gold_e[:, :uttr_len] - 1
        pred_e_cpu = pred_e.permute(0, 2, 1).cpu()
        gold_e_cpu = gold_e.cpu()
        # criterion = FocalLoss(reduction='mean', ignore_index=-1)
        criterion = FocalLoss()
        # criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)  # 为-1的值被忽略
        loss_e = criterion(pred_e_cpu, gold_e_cpu)
        return loss_e

    def loss_pre_cau(self, pred_c, gold_c):
        uttr_len = pred_c.size(1)
        gold_c = gold_c[:, :uttr_len]
        c_mask = gold_c != -1  # 为 -1 的位置为填充值
        pred_c = pred_c.squeeze(-1).masked_select(c_mask).cpu()
        true_c = gold_c.masked_select(c_mask).float().cpu()
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss_c = criterion(pred_c, true_c)
        # loss_c = focal_loss_bce(pred_c, true_c, alpha=1, gamma=2, reduction='mean')
        return loss_c

    def loss_pre_ec(self, couples_pred, emo_cau_pos, ec_pair, uttr_mask, test=False):
        couples_true, couples_mask = self.truncate_ec_pairs(couples_pred, emo_cau_pos, ec_pair, uttr_mask, test)
        couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
        couples_true = torch.FloatTensor(couples_true).to(DEVICE)
        couples_true = couples_true.masked_select(couples_mask)
        couples_pred = couples_pred.masked_select(couples_mask)
        loss_couple = focal_loss_bce(couples_pred, couples_true, alpha=1, gamma=2, reduction='mean')
        return loss_couple

    # def loss_pre_emo(self, pred_e, gold_e):
    #     uttr_len = pred_e.size(1)
    #     # 这里进行截断
    #     gold_e = gold_e[:,:uttr_len]
    #     gold_e = gold_e[:,:uttr_len]-1 # 计算时还是要从0-6
    #     # 该行代码可能导致计算的不确定性，影响可复现性
    #     # 将预测和标签移至CPU
    #     pred_e_cpu = pred_e.permute(0, 2, 1).cpu()
    #     gold_e_cpu = gold_e.cpu()
    #     criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1) # 为-1的值被忽略
    #     loss_e = criterion(pred_e_cpu, gold_e_cpu)
    #     return loss_e

    # def loss_pre_cau(self, pred_c, gold_c):
    #     uttr_len = pred_c.size(1)
    #     # 这里进行截断
    #     gold_c = gold_c[:,:uttr_len]
    #     c_mask = gold_c != -1  # 为 -1 的位置为填充值
    #     pred_c = pred_c.squeeze(-1).masked_select(c_mask).cpu()
    #     true_c = gold_c.masked_select(c_mask).float().cpu()
    #     criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #     loss_c = criterion(pred_c, true_c)
    #     return loss_c

    # def loss_pre_ec(self, couples_pred, emo_cau_pos, ec_pair, uttr_mask, test = False):
    #     couples_true, couples_mask = self.truncate_ec_pairs(couples_pred, emo_cau_pos, ec_pair, uttr_mask, test)
    #     couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
    #     couples_true = torch.FloatTensor(couples_true).to(DEVICE)
    #     criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #     couples_true = couples_true.masked_select(couples_mask)
    #     couples_pred = couples_pred.masked_select(couples_mask)
    #     loss_couple = criterion(couples_pred, couples_true)
    #     return loss_couple

    def truncate_ec_pairs(self, couples_pred, emo_cau_pos, ec_pair, uttr_mask, test=False):
        batch, n_couple = couples_pred.size()
        couples_true, couples_mask = [], []
        for i in range(batch):
            uttr_mask_i = uttr_mask[i]
            uttr_len = uttr_mask_i.sum().item()
            uttr_couples_i = ec_pair[i]  # 该对话的真实配对
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):  # 对每个笛卡尔积后的情绪原因配对（经过截断后的），但是经过batch的padding后，可能有一些配对是无效的
                if emo_cau[0] > uttr_len or emo_cau[1] > uttr_len:  # 情绪idx或者原因idx如果超出了该对话的长度
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in uttr_couples_i.tolist() else 0)
            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
        return couples_true, couples_mask


class EmotionMLP(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(EmotionMLP, self).__init__()
        self.emotion_mlp_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.emotion_mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence_outputs_h):
        em_hiddens_h = self.emotion_mlp_1(sequence_outputs_h)
        em_logits = self.emotion_mlp_2(sequence_outputs_h)
        return em_logits


class CauseMLP(nn.Module):
    def __init__(self, hidden_dim):
        super(CauseMLP, self).__init__()
        self.cause_mlp_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.cause_mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence_outputs_h):
        cause_hiddens_h = self.cause_mlp_1(sequence_outputs_h)
        ca_logits = self.cause_mlp_2(sequence_outputs_h)
        return ca_logits


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(in_channels=1, out_channels=input_size, kernel_size=1)
        self.w2 = nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w2(F.elu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

#     def forward(self, x):
#         residual = x
#         x = F.gelu(self.fc1(x))
#         x = self.fc2(x)
#         x = self.dropout(x)
#         x += residual
#         x = self.layer_norm(x)
#         return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, pred, target):
        logpt = -self.ce(pred, target)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad


class FocalLoss2(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, -1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


def focal_loss_bce(pred, target, alpha=1, gamma=2, reduction='mean'):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * ((1 - pt) ** gamma) * bce_loss
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss