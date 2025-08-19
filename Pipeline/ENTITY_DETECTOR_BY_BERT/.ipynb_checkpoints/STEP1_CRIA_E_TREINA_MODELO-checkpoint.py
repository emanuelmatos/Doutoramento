#!/usr/bin/env python
# coding: utf-8

# ## INPUT de  __INITIAL_NERS/OUTPUT_RESULTS/
# ####  ENTITY_DETECTOR_BY_BERT/INPUT_DATASET/decision_entity_bert_arrange.csv
# 

# In[1]:


#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import sys
from tqdm import tqdm, trange

import os


# In[2]:


import os

working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
#if parts[-1] == "pipeline":
if parts[-1].startswith("pipeline"): ### BUG SOLVED MAY 2025
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[3]:


from datetime import datetime
start_time = datetime.now()
# do your work here

# In[38]:
#base = "/home/ematos/phd/pt/INITIAL_NERS/"
sys.path.append(pipeline)

from config.configsite import * # basedir,inputdir,outputdir


# In[4]:


filesdir = f"{pipeline}/INITIAL_NERS/{resultsdir}/"


# In[5]:


outputdir=pipeline+"ENTITY_DETECTOR_BY_BERT/"+resultsdir+"/"
print(f"output dir = {outputdir}") # new AT april 2025


# In[16]:


input_folder = filesdir
#input_folder = '/home/ematos/phd/pt/ENTITY_DETECTOR_BY_BERT/INPUT_DATASET/'
input_file =  'decision_entity_bert_arrange.csv'

#output_folder = '/home/ematos/phd/pt/ENTITY_DETECTOR_BY_BERT/OUTPUT_RESULTS/'

model_name = "Bert_text_entity_19032025.h5"
model_dir = "Bert_text_entity_19032025.h5"
filename = "Bert_text_entity_19032025.h5"
model_file = outputdir+filename
model_directory = outputdir+model_dir
#-----------------------------------------------------------------------------------------
#from configsite import basedir,inputdir,outputdir

#direxit=basedir
#filesdir=basedir

#______________________________________________________________________________________

data1 = pd.read_csv(f'{input_folder}{input_file}',sep='\t', encoding="utf8").ffill()


data1.tail(10)

#data_all = pd.concat([data1,data2],ignore_index=True)

data_all = pd.concat([data1],ignore_index=True)

df=data_all
df['POS']=df['POSITION']


# In[7]:


df


# In[8]:


from sklearn.model_selection import train_test_split

# 1. Get unique FILE names
unique_files = df['FILE'].unique()

# 2. Split the file list
train_files, test_files = train_test_split(unique_files, test_size=0.3, random_state=42)

# 3. Create train/test DataFrames by filtering rows where FILE is in the respective split
df_train = df[df['FILE'].isin(train_files)].reset_index(drop=True)
df_test = df[df['FILE'].isin(test_files)].reset_index(drop=True)


# In[9]:


df_train.to_csv(input_folder+'decision_entity_bert_arrange_train.csv',index=None, sep="\t")
df_test.to_csv(input_folder+'decision_entity_bert_arrange_test.csv',index=None, sep="\t")


# In[10]:


data = df_train


# In[11]:


import random
random.seed(30)
sem=random.randint(1, 50)
print("first Number", sem)


data=data_all


# In[7]:


#################################################################
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
###############################################################
getter = SentenceGetter(data)
##########################################################
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
sentences[0]


# In[8]:


############################################################
labels = [[s[2] for s in sentence] for sentence in getter.sentences]
print(labels[0])


# In[9]:


##############################################################
tag_values = list(set(data["Tag"].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}
########################################################


# In[10]:


tag_values


# In[ ]:





# In[11]:


###################################################
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

#from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

torch.__version__


# In[12]:


MAX_LEN = 75
bs = 32
#####################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


# In[13]:


torch.cuda.get_device_name(0)


import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Check the number of GPUs available
n_gpu = torch.cuda.device_count()
print(f"Number of GPUs available: {n_gpu}")

# If GPUs are available, print their names
if n_gpu > 0:
    for i in range(n_gpu):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tokenizer

#####################################################
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
###########################################################################


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]
######################################################################
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
##############################################################################

######### Split Train / Test (10%)

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")
#####################################################################################
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
###############################################################################
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
###########################################################################
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=sem, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=sem, test_size=0.1)
print(sem)


##############################################################################
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
############################################################################
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

########################################################################
import transformers
#from transformers import BertForTokenClassification, AdamW
from transformers import BertForTokenClassification
#from transformers.optimization import AdamW
from torch.optim import AdamW

from transformers import DistilBertForSequenceClassification

print(transformers.__version__)
##########################################################################


# In[ ]:


##########################################################################
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)



model.cuda();


# In[ ]:


################################################################
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

# In[38]:
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)


##################################################################################
from transformers import get_linear_schedule_with_warmup

epochs = 100
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


################################################################################
from seqeval.metrics import f1_score, accuracy_score
##############################################################


x=train_dataloader.dataset.tensors[0]
type(x)
#x.dtype='Long'
x=x.to(dtype=torch.long)



x1=train_dataloader.dataset.tensors[1]
type(x1)
#x.dtype='Long'
x1=x1.to(dtype=torch.long)



x2=train_dataloader.dataset.tensors[2]
type(x2)
#x.dtype='Long'
x2=x2.to(dtype=torch.long)


train_dataloader.dataset.tensors=(x,x1,x2)



train_dataloader.dataset.tensors


valid_dataloader.dataset.tensors


x=valid_dataloader.dataset.tensors[0]
type(x)
#x.dtype='Long'
x=x.to(dtype=torch.long)


x1=valid_dataloader.dataset.tensors[1]
type(x1)
#x.dtype='Long'
x1=x1.to(dtype=torch.long)

x2=valid_dataloader.dataset.tensors[2]
type(x2)
#x.dtype='Long'
x2=x2.to(dtype=torch.long)



valid_dataloader.dataset.tensors=(x,x1,x2)


# In[ ]:


# In[38]:


## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []
a=100
for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    _=1
    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        #print(outputs)
        loss = outputs[0]
        print(".",end="")
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


       # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    b=eval_loss
    validation_loss_values.append(eval_loss)
    print(a,b)
    if(b>a):
        break
    else: 
        a=b
    print(a,b)
    print("Validation loss: {}".format(eval_loss))
    
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    
   #print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    
    print()

   
print(epochs)



# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

validation_loss_values


# In[17]:


print(f"Model File = {model_file}")
print(f"Model dir = {model_directory}")




model.save_pretrained(model_file)
tokenizer.save_pretrained(model_file)

model.save_pretrained(model_directory)
tokenizer.save_pretrained(model_directory)

# Save the tags to the file
with open(f"{model_directory}/tag.txt", "w") as f:
    for s in tag_values:
        f.write(str(s) + "\n")


# In[ ]:


##################################################################################
#######################  Visualize the training loss
##################################################################################
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns



model_file = f"{outputdir}/{filename}"
modeldir = f"{outputdir}/{model_dir}"
# Ensure the directory exists
print(f"Model dir = {modeldir}")
os.makedirs(modeldir, exist_ok=True)



# Use plot styling from seaborn.
sns.set(style='darkgrid')


# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams['figure.figsize'] = (5,5)

# Plot the learning curve.
plt.plot(loss_values, 'b-o', label="training loss")
plt.plot(validation_loss_values, 'r-o', label="validation loss")

# Label the plot.
plt.title("Learning curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


# Save the plot to a file.
output_file = f"{modeldir}/learning_curve.png"  # Replace with your desired path
plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")


plt.show()


# In[42]:


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[ ]:





# In[ ]:




