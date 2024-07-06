"""BERT Finetune"""
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Hyperparameters

settings = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "learning_rate": 2e-5,
    "epochs": 4,
    "batch_size": 32,
}

device = torch.device(settings["device"])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# Generate dataset

df = pd.read_csv(
    "./data/train.csv",
    delimiter=",",
    header=None,
    names=["content", "rating"]
)

inputs = []
attn_masks = []
targets = []

for _, row in df.iterrows():
    if row["rating"] == "rating": # Ignore header
        continue
    encoded_dict = tokenizer.encode_plus(
        row["content"],              # Sentence to encode.
        add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
        max_length = 128,              # Pad & truncate all sentences.
        truncation=True,
        pad_to_max_length = True,
        return_attention_mask = True, # Construct attn. masks.
        return_tensors = 'pt',        # Return pytorch tensors.
    )
    inputs.append(encoded_dict['input_ids'])
    attn_masks.append(encoded_dict['attention_mask'])
    targets.append(int(row["rating"]))

inputs = torch.cat(inputs, dim=0)
attn_masks = torch.cat(attn_masks, dim=0)

dataset = torch.utils.data.TensorDataset(torch.tensor(inputs), torch.tensor(attn_masks), torch.tensor(targets))
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=settings["batch_size"], shuffle=True)

# Set up model and optimizer

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # 12-layer BERT model, with an uncased vocab.
    num_labels = 2,      # 2 output labels for binary classification.
    output_attentions = False,
    output_hidden_states = False,
)

model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], eps=1e-8)

steps = len(train_dataloader) * settings["epochs"]
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)

# Function to estimate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop

training_stats = []

for epoch_i in range(0, settings["epochs"]):
    
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, settings["epochs"]))
    print('Training...')


    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode
    model.train()

    # For each batch of training data
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # Unpack this training batch from our dataloader
        b_inputs = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        b_targets = batch[2].to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        result = model(
            b_inputs,
            token_type_ids=None,
            attention_mask=b_attn_mask,
            labels=b_targets
        )

        loss = result[0]
        logits = result[1]

        # Accumulate training loss
        total_train_loss += loss.item()

        # Backpropogate
        loss.backward()

        # Clip the norm of the gradients to 1.0. to help prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()

        # Update learning rate
        scheduler.step()

    # Calculate the average loss over all of the batches
    avg_train_loss = total_train_loss / len(train_dataloader)            
    

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    # Validation after each epoch
    print("")
    print("Running Validation...")


    # Put the model in evaluation mode
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader
        b_inputs = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        b_targets = batch[2].to(device)
        
        with torch.no_grad():

            # Forward pass
            result = model(
                b_inputs,
                token_type_ids=None,
                attention_mask=b_attn_mask,
                labels=b_targets
            )
            loss = result[0]
            logits = result[1]
            
        # Accumulate the validation loss
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_targets.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
        }
    )

print("")
print("Training complete!")
print(training_stats)

# Save model

torch.save(model.state_dict(), "./model.pt")
