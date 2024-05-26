import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class PhoBertModel(torch.nn.Module):

    def __init__(self):

        super(PhoBertModel, self).__init__()

        self.phobert = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):
        
        output = self.phobert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        
        return output
    
unique_labels = ['I-JOB',
 'I-DATE',
 'I-AGE',
 'B-ORG',
 'B-PER',
 'I-MISC',
 'I-PER',
 'B-GENDER',
 'I-SYMPTOM_AND_DISEASE',
 'O',
 'B-DATE',
 'I-LOC',
 'B-PATIENT_ID',
 'B-SYMPTOM_AND_DISEASE',
 'B-JOB',
 'I-TRANSPORTATION',
 'B-MISC',
 'B-TRANSPORTATION',
 'I-PATIENT_ID',
 'B-AGE',
 'I-ORG',
 'I-GENDER',
 'B-LOC']

labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
ids_to_labels = {v: k for v, k in enumerate(unique_labels)}


TOKENIZER_PATH = "luzox/N_NER"
# test_md = torch.load('model.pt', map_location=torch.device('cpu'))
test_md = torch.load('model.pt')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def align_word_ids(text, flag):
    label_all_tokens = flag
    
    text = text.split()
  
    tokenized_inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, is_split_into_words=True)

    word_ids = tokenized_inputs.input_ids

    start_part = True
    label_ids = []
    count = 0
    
    for i in range(len(word_ids)):
        
        if word_ids[i] == 0 or word_ids[i] == 1 or word_ids[i] == 2:
            label_ids.append(-100)
            
        elif count < len(text) and ''.join(tokenizer.decode(tokenized_inputs['input_ids'][i]).split()) == text[count]:
            label_ids.append(1)
            count+=1
            start_part = True
        else:
            if start_part:
                label_ids.append(1)
                count+=1
                start_part = False
            else:
                label_ids.append(1 if label_all_tokens else -100)           
    return label_ids

def ner(model, sentence, flag_align_label):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    text = tokenizer(sentence, padding='max_length', max_length = 256, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence, flag_align_label)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    st.write(sentence)
    st.write(prediction_label)





st.title('Named Entity Recognization')
txt = st.text_input('Nhập tại đây')

if txt:
    ner(test_md,txt,flag_align_label=False)
