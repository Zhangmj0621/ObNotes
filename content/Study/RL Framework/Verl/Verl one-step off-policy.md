![[Pasted image 20260408135652.png]]
代码流程pipeline大致如下：

```python
# iterator generator, simplify one-step integration of the training process
def _create_continuous_iterator(self):
   for epoch in range(self.config.trainer.total_epochs):
      iterator = iter(self.train_dataloader)
      for batch_dict in iterator:
         yield epoch, batch_dict

# read next batch samples, parameters sync and launch asyn gen_seq
def _async_gen_next_batch(self, continuous_iterator):
   # read train_data
   try:
      epoch, batch_dict = next(continuous_iterator)
   except StopIteration:
      return None
   batch = DataProto.from_single_dict(batch_dict)
   gen_batch = batch_pocess(batch)
   # sync weights from actor to rollout
   self.sync_rollout_weights()
   # async generation
   gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)
   # future encapsulated
   return GenerationBatchFuture(epoch, batch, gen_batch_output)

continuous_iterator = self._create_continuous_iterator()
# run rollout first to achieve one-step-off
batch_data_future = self._async_gen_next_batch(continuous_iterator)

while batch_data_future is not None:
   # wait for the gen_seq result from the previous step
   batch = batch_data_future.get()
   # launch the next async call to generate sequences
   batch_data_future = self._async_gen_next_batch(continuous_iterator)

   # compute advantages 
   batch = critic.compute_values(batch)
   batch = reference.compute_log_prob(batch)
   batch = reward.compute_reward(batch)
   batch = compute_advantages(batch)

   # model update
   critic_metrics = critic.update_critic(batch)
   actor_metrics = actor.update_actor(batch)
```

目前来看，所谓的one-step off policy仍然单个batch内还是需要等待最长的request完成，无法完全异步的sample level的插入