import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertTokenizer, BertModel
import config
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraClassificationHead


class Models(nn.Module):
    def __init__(self, mdl):
        super(Models, self).__init__()
        self.model = AutoModel.from_pretrained(mdl)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # pooled_ouput is the pooler layer -- simply put the output from the CLS token
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).last_hidden_state
        output = self.drop(hidden[:,0,:])
        output = self.out(output)
        return output


class BertTrace(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = ElectraClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids,**kwargs):
        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        logits = self.cls(hidden)
        return logits

