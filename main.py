import torch
import torch.nn as nn
from transformers import GPT2Model, BertModel, T5Tokenizer, AutoTokenizer, BertTokenizer
from transformers import T5ForConditionalGeneration

# تعریف کلاس LNNM2
class LNNM2(nn.Module):
    def __init__(self, gpt2_model, flan_model):
        super(LNNM2, self).__init__()
        self.gpt2_model = gpt2_model
        self.flan_model = flan_model
        self.fc = nn.Linear(
            gpt2_model.config.hidden_size +
            flan_model.config.d_model,
            768
        )

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        gpt2_outputs = self.gpt2_model(input_ids=input_ids).last_hidden_state
        # ارسال ورودی‌های دیکودر به FLAN-T5
        flan_outputs = self.flan_model(input_ids=decoder_input_ids, decoder_attention_mask=attention_mask).last_hidden_state
        combined_features = torch.cat((gpt2_outputs, flan_outputs), dim=-1)
        output = self.fc(combined_features)
        return output

# تعریف کلاس LNNM3
class LNNM3(nn.Module):
    def __init__(self, lnnm2_model, bert_model):
        super(LNNM3, self).__init__()
        self.lnnm2_model = lnnm2_model
        self.bert_model = bert_model
        self.fc = nn.Linear(768 + self.bert_model.config.hidden_size, 768)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        lnnm2_outputs = self.lnnm2_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        combined_features = torch.cat((lnnm2_outputs, bert_outputs), dim=-1)
        output = self.fc(combined_features)
        return output

# بارگذاری مدل‌های پایه‌ای
gpt2_model = GPT2Model.from_pretrained('gpt2', cache_dir="cache")
bert_model = BertModel.from_pretrained('bert-base-uncased', cache_dir="cache")

# بارگذاری مدل FLAN-T5
flan_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small', cache_dir="cache")

# تعریف مدل LNNM2
lnnm2_model = LNNM2(gpt2_model, flan_model)

# بارگذاری مدل LNNM3
lnnm3_model = LNNM3(lnnm2_model, bert_model)
lnnm3_model.eval()

# آماده‌سازی ورودی
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="cache")
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir="cache")
t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small', cache_dir="cache")

# آماده‌سازی متن ورودی
input_text = "This is a sample input sentence for testing."
encoded_input = tokenizer(input_text, return_tensors='pt')

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

# ایجاد decoder_input_ids برای مدل FLAN-T5
# به طور معمول، decoder_input_ids برای تولید متن استفاده می‌شود
# در اینجا، برای تست می‌توانید از input_ids به عنوان decoder_input_ids استفاده کنید
# در عمل باید ورودی‌های مناسب برای دیکودر آماده کنید
decoder_input_ids = t5_tokenizer.encode(input_text, return_tensors='pt')

# اجرای مدل LNNM3
with torch.no_grad():
    output = lnnm3_model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

# تبدیل خروجی به متن
# توجه داشته باشید که باید ورودی‌ها به توکن‌ها تبدیل شوند
generated_ids = torch.argmax(output, dim=-1)  # انتخاب بیشترین احتمال

# تبدیل توکن‌ها به متن
generated_text = gpt2_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Generated text:", generated_text)
