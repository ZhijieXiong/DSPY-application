import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_dir, freeze_bert=False, dim_in=768, dim_out=20, pooling="max"):
        super(BERTClassifier, self).__init__()
        self.pooling = pooling
        self.bert = BertModel.from_pretrained(bert_model_dir)
        self.classifier = nn.Linear(dim_in, dim_out)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.classifier.apply(init_weights)

    def forward(self, input_ids, attention_mask, sub_token_list_lens):
        # input_ids: (bs, num_sub_seq, 512)
        input_ids = input_ids.reshape(-1, 512)
        attention_mask = attention_mask.reshape(-1, 512)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding_all = outputs[0][:, 0, :]

        batch_size = sub_token_list_lens.shape[0]
        max_sub_seq_num = int(cls_embedding_all.shape[0] / batch_size)
        cls_embedding = []
        for i in range(batch_size):
            sub_seq_num = sub_token_list_lens[i].item()
            cls_embedding_this_text = cls_embedding_all[i * max_sub_seq_num: i * max_sub_seq_num + sub_seq_num]
            if self.pooling == "max":
                cls_embedding_pool, _ = torch.max(cls_embedding_this_text, dim=0)
            elif self.pooling == "mean":
                cls_embedding_pool, _ = torch.mean(cls_embedding_this_text, dim=0)
            else:
                cls_embedding_pool, _ = torch.max(cls_embedding_this_text, dim=0)
            cls_embedding.append(cls_embedding_pool)
        logits = self.classifier(torch.stack(cls_embedding))

        return logits
