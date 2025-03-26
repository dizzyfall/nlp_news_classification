'''
@Project ：NLPNewsClassification
@File    ：pretrain_bert.py
@Author  ：DZY
@Date    ：2025/3/14 12:38
'''

from torch import nn

from src.utils import timer, checkpoints


def load_pretrain_checkpoint(model, optimizer, device, checkpoint_relative_path=None):
    if checkpoint_relative_path is None:
        step = 0
        total_mlm_loss = 0.0
        total_processed_samples = 0.0
        cum_time_list = []
        return step, total_mlm_loss, total_processed_samples, cum_time_list
    step, total_mlm_loss, total_processed_samples, cum_time_list = checkpoints.load_checkpoint(model,
                                                                                               checkpoint_relative_path,
                                                                                               device, optimizer)
    return step, total_mlm_loss, total_processed_samples, cum_time_list


class PretrainBERT():
    def __init__(self, current_step=0, total_mlm_loss=0.0, total_processed_samples=0.0, cum_time_list=None):
        if cum_time_list is None:
            cum_time_list = []
        self.current_step = current_step
        self.total_mlm_loss = total_mlm_loss
        self.total_processed_samples = total_processed_samples
        self.cum_time_list = cum_time_list

    def _get_bert_batch_loss(self, net, loss, vocab_size, tokens_X, segments_X, valid_lens_X, pred_positions_X,
                             mlm_weights_X, mlm_Y):
        _, mlm_Y_hat, _ = net(tokens_X, segments_X, valid_lens_X, pred_positions_X)
        mlm_loss = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
        mlm_loss = mlm_loss.sum() / (mlm_weights_X.sum() + 1e-8)
        return mlm_loss

    def pretrain(self, pretrain_iter, net, loss, vocab_size, pretrain_optimizer, pretrain_scheduler,
                 num_pretrain_iter_steps,
                 checkpoints_relative_path, devices):
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
        pretrain_timer = timer.Timer()
        num_pretrain_iter_steps_reached = False
        save_checkpoint_interval = 1000
        pretrain_info_interval = 100

        print("start pretraining...")
        while self.current_step < num_pretrain_iter_steps and not num_pretrain_iter_steps_reached:
            for tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X, mlm_Y in pretrain_iter:
                tokens_X = tokens_X.to(devices[0])
                segments_X = segments_X.to(devices[0])
                valid_lens_X = valid_lens_X.to(devices[0])
                pred_positions_X = pred_positions_X.to(devices[0])
                mlm_weights_X = mlm_weights_X.to(devices[0])
                mlm_Y = mlm_Y.to(devices[0])

                pretrain_timer.start()
                pretrain_optimizer.zero_grad()
                mlm_loss = self._get_bert_batch_loss(net, loss, vocab_size, tokens_X, segments_X, valid_lens_X,
                                                     pred_positions_X, mlm_weights_X, mlm_Y)
                mlm_loss.backward()
                pretrain_optimizer.step()

                if pretrain_scheduler is not None:
                    pretrain_scheduler.step()
                pretrain_timer.stop()

                self.total_mlm_loss += mlm_loss.item()
                self.total_processed_samples += tokens_X.shape[0]
                self.cum_time_list = pretrain_timer.get_cumulate_time()

                if (self.current_step + 1) % pretrain_info_interval == 0:
                    print(
                        f"Iter Steps: {self.current_step + 2 - pretrain_info_interval}-{self.current_step + 1} ---- "
                        f"Avg MLM Loss: {self.total_mlm_loss / (self.current_step + 1):.4f} ---- "
                        f"Cumulative Iter Time: {self.cum_time_list[self.current_step]:.4f} sec")

                if (self.current_step + 1) % save_checkpoint_interval == 0:
                    checkpoint_dir_name = f"checkpoint_step_{self.current_step + 1}"
                    checkpoint_file_name = f"checkpoint_step_{self.current_step + 1}.pth"
                    checkpoint_pretrain_info_tuple = (
                        self.total_mlm_loss, self.total_processed_samples, self.cum_time_list)
                    checkpoints.save_pretrain_checkpoint(net, pretrain_optimizer,
                                                         self.current_step + 1, checkpoint_pretrain_info_tuple,
                                                         checkpoints_relative_path,
                                                         checkpoint_dir_name, checkpoint_file_name)

                self.current_step += 1

                if self.current_step == num_pretrain_iter_steps:
                    num_pretrain_iter_steps_reached = True
                    break

        pretrain_total_time = pretrain_timer.get_total_time()
        print(f"Pretrain BERT Total Time: {timer.format_duration(pretrain_total_time)}")
        print(f"Pretrain BERT Total Avg MLM Loss: {self.total_mlm_loss / self.current_step:.4f}")
        print(f"Pretrain BERT Total Num Processed Samples: {self.total_processed_samples}")
        print(
            f"Pretrain BERT Processing Samples Speed: {self.total_processed_samples / pretrain_total_time:.4f} samples/sec")
