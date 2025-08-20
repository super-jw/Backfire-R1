import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from prompt import *
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import random
from openai import OpenAI
import numpy as np

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids
    
    def ensure_prefix_before_target(self, new_input_ids, target=128006, prefix=128009, pad_value=128009):
        if not isinstance(new_input_ids, torch.Tensor):
            raise ValueError("输入必须是PyTorch张量")
        
        # 获取批次大小和序列长度
        batch_size, seq_len = new_input_ids.shape
        processed_rows = []
        
        for i in range(batch_size):
            # 将当前行转换为列表以便插入操作
            row = new_input_ids[i].tolist()
            # 从后往前遍历，避免插入元素后影响后续索引
            j = len(row) - 1
            
            while j >= 0:
                if row[j] == target:
                    # 检查前一个元素是否为prefix
                    if j == 0 or row[j-1] != prefix:
                        # 插入prefix到target前面
                        row.insert(j, prefix)
                j -= 1
                
            processed_rows.append(row)
        
        # 找到处理后最长的行长度，用于统一长度
        max_length = max(len(row) for row in processed_rows)
        
        # 填充所有行到相同长度
        padded_rows = []
        for row in processed_rows:
            # 计算需要填充的长度
            pad_length = max_length - len(row)
            # 填充并添加到结果列表
            padded_row = row + [pad_value] * pad_length
            padded_rows.append(padded_row)
        
        # 转换回张量并保持原始数据类型
        return torch.tensor(padded_rows, dtype=new_input_ids.dtype)

    def ensure_prefix_before_target_2(self, responses, responses_with_info_mask, target=128006, prefix=128009, pad_value=128009):
        if not isinstance(responses, torch.Tensor):
            raise ValueError("输入必须是PyTorch张量")
        
        # 获取批次大小和序列长度
        batch_size, seq_len = responses.shape
        processed_rows = []
        processed_rows_with_info_mask = []
        
        for i in range(batch_size):
            # 将当前行转换为列表以便插入操作
            row = responses[i].tolist()
            row_with_info_mask = responses_with_info_mask[i].tolist()
            # 从后往前遍历，避免插入元素后影响后续索引
            j = len(row) - 1
            
            while j >= 0:
                if row[j] == target:
                    # 检查前一个元素是否为prefix
                    if j == 0 or row[j-1] != prefix:
                        # 插入prefix到target前面
                        row.insert(j, prefix)
                        row_with_info_mask.insert(j, prefix)
                j -= 1
                
            processed_rows.append(row)
            processed_rows_with_info_mask.append(row_with_info_mask)
        
        # 找到处理后最长的行长度，用于统一长度
        max_length = max(len(row) for row in processed_rows)
        
        # 填充所有行到相同长度
        padded_rows = []
        for row in processed_rows:
            # 计算需要填充的长度
            pad_length = max_length - len(row)
            # 填充并添加到结果列表
            padded_row = row + [pad_value] * pad_length
            padded_rows.append(padded_row)
        
        max_length = max(len(row) for row in processed_rows_with_info_mask)
        
        # 填充所有行到相同长度
        padded_rows_with_info_mask = []
        for row in processed_rows_with_info_mask:
            # 计算需要填充的长度
            pad_length = max_length - len(row)
            # 填充并添加到结果列表
            padded_row = row + [pad_value] * pad_length
            padded_rows_with_info_mask.append(padded_row)
        
        # 转换回张量并保持原始数据类型
        return torch.tensor(padded_rows, dtype=responses.dtype), torch.tensor(padded_rows_with_info_mask, dtype=responses.dtype)



    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor, eot_token_num: int) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids,
        ], eot_token_num=eot_token_num)

        model_type = 'llama'
        if model_type == 'llama':
            new_input_ids = self.ensure_prefix_before_target(new_input_ids)
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids, eot_token_num=eot_token_num, model_type=model_type)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        new_rollings.non_tensor_batch.update(rollings.non_tensor_batch)
        
        return new_rollings
    
    def modify_mask(self, mask, eot_token_num):
        n, m = mask.shape
        modified_mask = mask.clone()  # 避免修改原始张量
        
        for i in range(n):
            # 获取当前行中为False的元素索引
            false_indices = torch.where(~modified_mask[i])[0]
            
            # 如果False元素数量不少于2，则修改最后两个
            if len(false_indices) >= eot_token_num:
                indices_to_change = false_indices[-eot_token_num:]
                modified_mask[i, indices_to_change] = True
            # 如果False元素只有1个，则只修改这一个
            elif len(false_indices) == 1:
                modified_mask[i, false_indices[0]] = True
        
        return modified_mask
    
    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True,
                eot_token_num: int = 0
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          eot_token_num: int = 0) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False,
                    eot_token_num=eot_token_num
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        model_type = 'llama'
        if model_type == 'llama':
            responses, responses_with_info_mask = self.ensure_prefix_before_target_2(responses, responses_with_info_mask)
        effective_len = self.tensor_fn.create_attention_mask(responses, eot_token_num-2, model_type, begin=True).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        eot_token_num = 2
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # discuss with persuader
            if step == self.config.max_turns - 2:
                last_turn = True
            else:
                last_turn = False
            
            if step != self.config.max_turns - 1:
                persuader_responses_str, next_with_chat_template, new_discussion_history = self.execute_persuader(responses_str, rollings.non_tensor_batch['proposal'], rollings.non_tensor_batch['discussion_history'], last_turn=last_turn)
                # next_obs = ['<|im_end|>\n' + self.tokenizer.apply_chat_template(each_next_with_chat_template, add_generation_prompt=True, tokenize=False).split('<|im_end|>\n')[1] + '<|im_end|>\n<|im_start|>assistant\n' for each_next_with_chat_template in next_with_chat_template]
                next_obs = ['<|eot_id|><|start_header_id|>user<|end_header_id|>' + self.tokenizer.apply_chat_template(each_next_with_chat_template, add_generation_prompt=True, tokenize=False).split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[1] for each_next_with_chat_template in next_with_chat_template]
            else:
                for i in range(len(new_discussion_history)):
                    new_discussion_history[i] = np.append(new_discussion_history[i], ({"role": "assistant",
                                                                                       "content": responses_str[i]}))
                # next_obs = ['<|im_end|>\n' for i in range(len(rollings.non_tensor_batch['discussion_history']))]
                next_obs = ['<|eot_id|>' for i in range(len(rollings.non_tensor_batch['discussion_history']))]
            eot_token_num += 2
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Execute in environment and process observations
            # next_obs, dones, valid_action, is_search = self.execute_predictions(
            #     responses_str, self.tokenizer.pad_token, active_mask
            # )
            
            # curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            # active_mask = active_mask * curr_active_mask
            # active_num_list.append(active_mask.sum().item())
            # turns_stats[curr_active_mask] += 1
            # valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            # valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings.non_tensor_batch['discussion_history'] = new_discussion_history
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids,
                eot_token_num,
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                eot_token_num,
            )
            
        # final LLM rollout
        # if active_mask.sum():
        #     rollings.batch = self.tensor_fn.cut_to_effective_len(
        #         rollings.batch,
        #         keys=['input_ids', 'attention_mask', 'position_ids']
        #     )

        #     # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
        #     rollings_active = DataProto.from_dict({
        #         k: v[active_mask] for k, v in rollings.batch.items()
        #     })            
        #     gen_output = self._generate_with_gpu_padding(rollings_active)

        #     meta_info = gen_output.meta_info            
        #     responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
        #     responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
        #     # # Execute in environment and process observations
        #     _, dones, valid_action, is_search = self.execute_predictions(
        #         responses_str, self.tokenizer.pad_token, active_mask, do_search=False
        #     )

        #     curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
        #     active_mask = active_mask * curr_active_mask
        #     active_num_list.append(active_mask.sum().item())
        #     valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
        #     valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

        #     original_right_side = self._update_right_side(
        #         original_right_side,
        #         responses_ids,
        #     )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, eot_token_num, model_type='llama')

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            eot_token_num: int = 0,
                            model_type: str = 'llama') -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids'], model_type=model_type, eot_token_num=2, begin=False),
            self.tensor_fn.create_attention_mask(final_output['responses'], model_type=model_type, eot_token_num=eot_token_num-2, begin=True)
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids'], model_type=model_type, eot_token_num=2, begin=False),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'], model_type=model_type, eot_token_num=0, begin=True)
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_persuader(self, responses: List[str], proposal_list: List[str], discussion_history: List[str], text_only=True, last_turn=False) -> List[str]:
        persuader_system_prompt = [persuader_sys_prompt.replace("<proposal_text>", proposal) for proposal in proposal_list]
        messages = [[{"role": "system", "content": each_sys_prompt}] for each_sys_prompt in persuader_system_prompt]
        new_discussion_history = [each for each in discussion_history]
        for i in range(len(discussion_history)):
            each_discussion = discussion_history[i]
            for each_sentence in each_discussion[1:]:
                messages[i].append({"role": "user" if each_sentence["role"] == "assistant" else "assistant",
                                    "content": each_sentence["content"].split('<回答>')[1] if each_sentence["role"] == 'assistant' and '<回答>' in each_sentence["content"] else each_sentence['content']})
            messages[i].append({"role": "user",
                                "content": responses[i].split('<回答>')[1] if '<回答>' in responses[i] else responses[i]})
            new_discussion_history[i] = np.append(new_discussion_history[i], ({"role": "assistant", 
                                                                               "content": responses[i]}))

        model = random.choice(['Qwen2.5-72B-Instruct'])
        client = OpenAI(base_url='http://10.1.0.55:8088/v1', api_key='empty')
        if last_turn:
            results = [make_decision_prompt for message in messages]
        else:
            results = [client.chat.completions.create(
                            model=model,
                            messages=message,
                        ) for message in messages]
            if text_only:
                results = [result.choices[0].message.content for result in results]
        
        # next_with_chat_template = [each_discussion[-1:] for each_discussion in messages]
        next_with_chat_template = [[] for each_discussion in messages]
        for i in range(len(next_with_chat_template)):
            # if next_with_chat_template[i][0]["role"] == "user":
            #     next_with_chat_template[i][0]["role"] = "assistant"
            next_with_chat_template[i].append({"role": "user",
                                               "content": results[i]})
            new_discussion_history[i] = np.append(new_discussion_history[i], ({"role": "user",
                                                                               "content": results[i]}))
            
        return results, next_with_chat_template, new_discussion_history
        

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=False) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
