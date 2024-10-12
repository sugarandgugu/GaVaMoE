
prompt_template = {
    'instruction': 'You are a professional explainer, Your assignment involves providing users with a detailed explanation regarding a specific item.',
    'input'      : 'The user has a {} experience with the item, the item has {} features, please provide an explanation for recommending {} to {}.'
}
# 
class Prompt_Process:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.MAX_LENGTH = max_seq_length
    def __call__(self,examples):
        input_ids, attention_mask, labels = [], [], []
        
        mood = 'positive' if examples['rating'] >= 3 else 'negative' 
        feature = examples['feature']
        user = examples['user']
        item = examples['item']
        
        inputs = prompt_template['input'].format(mood, feature, 'ITEM', 'USER')
        instruction = self.tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_template['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{inputs}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  
        output = self.tokenizer(f"{examples['text']}<|eot_id|>", add_special_tokens=False)

        input_ids = instruction["input_ids"] + output["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + output["attention_mask"] + [1]  
        # instruction + input mask -100
        labels = [-100] * len(instruction["input_ids"]) + output["input_ids"] + [self.tokenizer.pad_token_id]  

        # truncation
        if len(input_ids) > self.MAX_LENGTH:  
            input_ids = input_ids[:self.MAX_LENGTH]
            attention_mask = attention_mask[:self.MAX_LENGTH]
            labels = labels[:self.MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }