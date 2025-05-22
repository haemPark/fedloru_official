import os
import json
import copy
import torch
import numpy as np
import random
import logging

from tqdm import tqdm
from collections import ChainMap, defaultdict

from src import MetricManager, CosineAnnealingWarmupRestarts
from .server import FedHMServer
from .client import FedHMClient

logger = logging.getLogger(__name__)

class FedHM():
    def __init__(self, args, server_dataset, client_datasets):
        super(FedHM, self).__init__()

        self.args = args
        self.round = 0
        self.curr_lr = self.args.lr
        self.rank_factor = self.args.rank_factor

        self.server_dataset = server_dataset
        self.client_datasets = client_datasets
        self.results = defaultdict(dict) # logging results container

        # Create Server & Clients
        self.server = self._create_server()
        self.clients = self._create_clients()

    def train(self):
        global_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=self.args.first_cycle_steps,
            cycle_mult=self.args.cycle_mult,
            max_lr=self.args.lr,
            min_lr=self.args.min_lr,
            warmup_steps=self.args.warmup_steps,
            gamma=self.args.cos_gamma
        )

        for curr_round in range(1, self.args.R+1):
            self.round = curr_round
            self.server.round = curr_round
            self.curr_lr = global_scheduler.get_lr()

            participating_clients = self._sample_clients()
            
            self._upload_factorized_model_to_clients(participating_clients)

            updated_sizes = self._request_local_update(participating_clients)

            recovered_state_dicts = self._request_recover_local_models(participating_clients)
            self.server.fullrank_model = self._request_global_aggregate(recovered_state_dicts, updated_sizes, participating_clients)

            if (curr_round % self.args.eval_every == 0) or (curr_round == self.args.R):
                self.evaluate(excluded_ids=participating_clients)
            global_scheduler.step()
            
        self.finalize()
            
    def _create_server(self):
        server = FedHMServer(args=self.args, server_dataset=self.server_dataset)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] Successfully Created the Server')

        return server
    
    def _create_clients(self):
        def __create_client(identifier, datasets):
            client = FedHMClient(args=self.args, training_set=datasets[0], test_set=datasets[-1], id=identifier)
            return client
        
        clients = []
        for identifier, datasets in enumerate(self.client_datasets):
            clients.append(__create_client(identifier, datasets))
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] Sucessfully created {self.args.K} clients!')
        
        return clients

    def _sample_clients(self, exclude=[]):
        if exclude == []: # Update - randomly select max(floor(C * K), 1) clients
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))
        else: # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                sampled_client_ids = sorted(random.sample([identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude], num_sampled_clients))
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Sample {num_sampled_clients} clients!')
        return sampled_client_ids

    def _upload_factorized_model_to_clients(self, participating_client_ids):
        factorized_model = self.server.get_factorized_lowrank_state_dict(self.rank_factor)
        for id in participating_client_ids:
            client = self.clients[id]
            client.download(factorized_model)
        
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Factorize the global model!')

    def _request_local_update(self, participating_client_ids):
        def __update_client(client, pbar, device=None):
            update_result = client.update(pbar, device)
            return {client.id: len(client.train_dataloader)}, {client.id: update_result}
        
        # 1) broadcast global model to local clients and synchronize learning-rate 
        for id in participating_client_ids:
            client = self.clients[id]
            client.args.lr = self.curr_lr        
        
        # 2) perform local clients' updates
        results = []
        with tqdm(
            total = sum([len(self.clients[id].train_dataloader) * self.args.E for id in participating_client_ids]),
            desc="ROUND {} Local Training".format(self.round)
        ) as pbar:        
            for id in participating_client_ids:
                # request update to selected clients
                results.append(__update_client(self.clients[id], pbar, self.args.device))

        # 3) record local training results
        update_sizes, _update_results = list(map(list, zip(*results)))
        update_sizes, _update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*_update_results))
        self.results[self.round]['clients_updated'] = self._log_results(
            update_sizes, 
            _update_results, 
            eval=False, 
            participated=True,
            save_raw=False
        )
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed updates of {"all" if participating_client_ids is None else len(participating_client_ids)} clients!')
        return update_sizes

    def _request_local_evaluate(self, ids, participated, retain_model, save_raw):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request evaluation to {"all" if ids is None else len(ids)} clients!')
        
        def __evaluate_client(client):
            client.lowrank_model.load_state_dict(
                self.server.get_factorized_lowrank_state_dict(client.lowrank_model, self.rank_factor)
            )
            eval_result = client.evaluate()
            if not retain_model:
                client.lowrank_model = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}
        
        if self.args.test_size == 0:
            print("!!!!!! NO TEST DATA : TEST_SIZE = 0")
            return None
        results = [__evaluate_client(self.clients[idx]) for idx in ids]
        _eval_sizes, _eval_results = list(map(list, zip(*results)))
        _eval_sizes, _eval_results = dict(ChainMap(*_eval_sizes)), dict(ChainMap(*_eval_results))

        self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
            _eval_sizes, 
            _eval_results, 
            eval=True, 
            participated=participated,
            save_raw=save_raw
        )
        return None

    def _request_recover_local_models(self, participating_client_ids):
        recovered_state_dict = defaultdict(dict)
        fullrank_model = self.server.fullrank_model
        for id in participating_client_ids:
            client = self.clients[id]
            recovered_state_dict[id] = client.recover_lowrank_model(fullrank_model)
        return recovered_state_dict

    def _request_global_aggregate(self, recovered_state_dicts, updated_sizes, participating_client_ids):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request Server Aggregation')

        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: float(nuemrator / sum(updated_sizes.values())) for identifier, nuemrator in updated_sizes.items()}

        global_model = self.server.fullrank_model
        for key in global_model.state_dict().keys():
            temp = torch.zeros_like(global_model.state_dict()[key])
            
            for id in participating_client_ids:
                temp += coefficients[id] * recovered_state_dicts[id][key]

            global_model.state_dict()[key].data.copy_(temp)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Finish Server Aggregation')
        return global_model
    
    def evaluate(self, excluded_ids):
        if self.args.eval_type != 'global': # `local` or `both`: evaluate on selected clients' holdout set
            selected_ids = self._sample_clients(exclude=excluded_ids)
            _ = self._request_local_evaluate(selected_ids, participated=True, retain_model=True, save_raw=self.round == self.args.R)
        if self.args.eval_type != 'local': # `global` or `both`: evaluate on the server's global holdout set 
            self._central_evaluate()
    
    @torch.no_grad()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        global_model = self.server.fullrank_model
        global_model.eval()
        global_model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = global_model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            global_model.to('cpu')
            mm.aggregate(len(self.server_dataset))

        # log result
        result = mm.results
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## loss
        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '
        
        ## metrics
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        self.results[self.round]['server_evaluated'] = result
       
    # DONE
    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            # get sample size
            num_samples.append(resulting_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()
        
        top10_indices = np.argpartition(losses_array, -int(0.1 * len(losses_array)))[-int(0.1 * len(losses_array)):] if len(losses_array) > 1 else 0
        top10 = np.atleast_1d(losses_array[top10_indices])
        top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

        bot10_indices = np.argpartition(losses_array, max(1, int(0.1 * len(losses_array)) - 1))[:max(1, int(0.1 * len(losses_array)))] if len(losses_array) > 1 else 0
        bot10 = np.atleast_1d(losses_array[bot10_indices])
        bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float), 
            'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
            'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        # metrics
        for name, val in metrics.items():
            val_array = np.array(val).astype(float)
            weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()
            
            top10_indices = np.argpartition(val_array, -int(0.1 * len(val_array)))[-int(0.1 * len(val_array)):] if len(val_array) > 1 else 0
            top10 = np.atleast_1d(val_array[top10_indices])
            top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

            bot10_indices = np.argpartition(val_array, max(1, int(0.1 * len(val_array)) - 1))[:max(1, int(0.1 * len(val_array)))] if len(val_array) > 1 else 0
            bot10 = np.atleast_1d(val_array[bot10_indices])
            bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

            total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
            result_dict[name] = {
                'avg': weighted.astype(float), 'std': std.astype(float), 
                'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
                'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
            }
                
            if save_raw:
                result_dict[name]['raw'] = val
        
        # log total message
        logger.info(total_log_string)
        return result_dict

    
    def finalize(self):
        """
            Save results.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Save results and the global model checkpoint!')
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file: # save results
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)
        torch.save(self.server.global_model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt')) # save model checkpoint
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...finished federated learning!')
