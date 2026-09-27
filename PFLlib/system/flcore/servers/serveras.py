import os
import json
import time
import copy
import numpy as np
# from flcore.clients.clientavg import clientAVG
from flcore.clients.clientas import clientAS
from flcore.servers.serverbase import Server


class FedAS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.fim_norm = getattr(args, 'fedas_fim_norm', 'proportional')
        self.use_pa = not getattr(args, 'fedas_no_pa', False)
        self.use_cs = not getattr(args, 'fedas_no_cs', False)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAS)

        print(f"\nFedAS modules: PA={self.use_pa}, CS={self.use_cs}, FIM norm={self.fim_norm}")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.print_shard_composition()
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def all_clients(self):
        return self.clients

    def send_selected_models(self, selected_ids, epoch):
        assert (len(self.clients) > 0)

        # for client in self.clients:
        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            progress = epoch / self.global_rounds
            
            client.set_parameters(copy.deepcopy(self.global_model), progress)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)    
    
    def aggregate_wrt_fisher(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        # calculate the aggregrate weight with respect to the FIM value of model
        FIM_weight_list = []
        for id in self.uploaded_ids:
            FIM_weight_list.append(self.clients[id].fim_trace_history[-1])

        raw_FIM_list = list(FIM_weight_list)
        FIM_weight_list = self.normalize_fim_weights(FIM_weight_list)
        self.print_aggregation_weights(raw_FIM_list, FIM_weight_list)

        for w, client_model in zip(FIM_weight_list, self.uploaded_models):
            self.add_parameters(w, client_model)

    def normalize_fim_weights(self, fim_values):
        """Map raw t-FIM traces to aggregation weights summing to 1 (E9)."""
        values = np.asarray(fim_values, dtype=np.float64)

        if self.fim_norm == 'softmax':
            # shift by the max for numerical stability, and scale by the mean
            # magnitude so the exponent stays in a usable range across datasets
            scale = np.mean(np.abs(values)) or 1.0
            shifted = (values - values.max()) / scale
            weights = np.exp(shifted)
        elif self.fim_norm == 'minmax':
            spread = values.max() - values.min()
            weights = (values - values.min()) / spread if spread > 0 else np.ones_like(values)
        else:  # proportional, the paper's Eq. 7
            weights = values

        total = weights.sum()
        if total <= 0:
            # every uploaded client had a degenerate t-FIM; fall back to uniform
            print('[FedAS] WARNING: FIM weights sum to 0, falling back to uniform weights.')
            return [1.0 / len(weights)] * len(weights)
        return list(weights / total)

    def print_aggregation_weights(self, raw_FIM_list, FIM_weight_list):
        """E7: per-client aggregation weight for the current round.

        Clients are tagged O (online cohort) or S (straggler) when -sc is active, and the
        two groups' mean weights are reported separately -- that contrast is the quantity
        Figure 4a plots. See plan section 7.14: the comparison is confounded by shard
        composition, so read it alongside the eff-class table printed at startup.
        """
        stragglers = set(self.straggler_ids)
        pairs = []
        for cid, w, raw in zip(self.uploaded_ids, FIM_weight_list, raw_FIM_list):
            tag = ('S' if cid in stragglers else 'O') if self.straggler_cohort else ''
            pairs.append(f"{cid}{tag}:{w:.4f}(raw {raw:.1f})")
        print(f"Agg weights [{self.fim_norm}] : " + '  '.join(pairs))

        if not self.straggler_cohort:
            return

        online_w = [w for cid, w in zip(self.uploaded_ids, FIM_weight_list)
                    if cid not in stragglers]
        strag_w = [w for cid, w in zip(self.uploaded_ids, FIM_weight_list)
                   if cid in stragglers]
        summary = f"  online n={len(online_w)} mean={np.mean(online_w):.4f}" if online_w else "  online n=0"
        summary += (f"  straggler n={len(strag_w)} mean={np.mean(strag_w):.4f}"
                    if strag_w else "  straggler n=0 (none selected this round)")
        print("Agg weight means :" + summary)

    def print_shard_composition(self):
        """Label-distribution stats per client, so E7's confound is visible in the log."""
        cfg_path = os.path.join('../dataset', self.dataset, 'config.json')
        if not os.path.exists(cfg_path):
            return
        with open(cfg_path) as f:
            shards = json.load(f)['Size of samples for labels in clients']

        stragglers = set(self.straggler_ids)
        print("\nShard composition (eff classes = exp(entropy) of label distribution):")
        for cid, shard in enumerate(shards[:self.num_clients]):
            counts = np.array([n for _, n in shard], dtype=np.float64)
            p = counts / counts.sum()
            eff = float(np.exp(-(p * np.log(p)).sum()))
            tag = ('straggler' if cid in stragglers else 'online   ') if self.straggler_cohort else ''
            print(f"  client{cid:<3} {tag} samples={int(counts.sum()):<6} "
                  f"classes={len(shard):<4} eff_classes={eff:.2f}")

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()

            selected_ids = [client.id for client in self.selected_clients]


            # self.send_models()

            # evaluate personalized models, ie FedAvg-C
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # self.send_models()
            self.send_selected_models(selected_ids, i)

            # print(f'send selected models done')

            # for client in self.selected_clients:
            #     client.train()
        

            for client in self.alled_clients:
                # print("===============")
                client.train(client.id in selected_ids)
            # assert 1==0


            self.print_fim_histories()



            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)


            if self.use_cs:
                self.aggregate_wrt_fisher()
            else:
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        print(f'+++++++++++++++++++++++++++++++++++++++++')

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAS)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def print_fim_histories(self):
        avg_fim_histories = []

        # Print FIM trace history for each client
        # for client in self.selected_clients:
        for client in self.alled_clients:
            formatted_history = [f"{value:.1f}" for value in client.fim_trace_history]
            print(f"Client{client.id} : {formatted_history}")
            avg_fim_histories.append(client.fim_trace_history)

        # Calculate and print average FIM trace history across clients
        avg_fim_histories = np.mean(avg_fim_histories, axis=0)
        formatted_avg = [f"{value:.1f}" for value in avg_fim_histories]
        print(f"Avg Sum_T_FIM : {formatted_avg}")

