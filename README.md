# Artificial-Judge
Artifical Judge attempts to use the BERT NLP pre-trained model to predict the winners of supreme court cases. This notebook uses [this kaggle dataset](https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction)
which can be found in the justice.csv file in this repo. 

## Setup
To begin, install conda or miniconda for your machine: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html      
Install jupyter lab as well: 
```
conda install jupyterlab -c conda-forge
```
Next, create a conda env to run this notebook in: 
```
conda create --name myenv python=3.7
```
run `conda activate myenv` to enter in to the env. Note: run `conda deactivate` to go back to base

Prepare a kernel for jupyter lab in your environment:       
- Activate the environment: `source activate myEnv`       
- Install ipykernel: `conda install ipykernel`     
- Run this command: `python -m ipykernel install --user --name myEnv --display-name “my_environment”`

Check import in the notebook for all necessary packages. You can check to see what packages are installed in your current conda env with `conda list`. If you are missing a package, simply run `conda install packagename`

## Fine Tuning
### Vailla Pytorch
For Vanilla Pytorch see section 3.1.1 and see the following code to tune some hyerparameters
```python
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = 16
        )
        
...

model = BertForSequenceClassification.from_pretrained(
    'bert-large-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )
```
or edit `num_epochs` before the training loop in 3.1.2

## Transformers Trainer
See section 3.2.3 "Fine Tuning":
```python
#Load pretrained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                      num_labels=2, 
                                                      #hidden_dropout_prob=0.4,
                                                      #attention_probs_dropout_prob=0.4
                                                      )

training_args = TrainingArguments(
    output_dir="test_trainer", 
    logging_dir='logs', 
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    logging_steps=50,
)
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
```
