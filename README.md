# Language-Module
## Requirements
Anaconda 3 required
### Installation steps
1. Download and install Anaconda 3
2. Navigate to the Language-Module folder in command prompt
3. Run the command `conda env create -n "name of environment" --file environment.yml` to install the environment
4. type `conda activate "name of environment"` to use the program

## Usage
This program trains transformers using the HuggingFace API<br><br>
### Tasks
st-en: Translation task<br>
en-st: Translation task<br>
ner-st: Translation task<br>
nl-ner: Named Entity Recognition task<br><br>

num_epochs=10 (default) <br>
mode=test or train (runs a test with 10% of data and smaller batch size)<br><br>
decrease batch_size for smaller machines (batch_size is in the `get_training_args` method inside the NERTrainer and TranslationTrainer classes<br>

### CLI
Use -h or --help for help<br><br>
Argument list:<br>
* --task (required), --model_checkpoint (required), --dataset_name (required), 
--model_name (required), --mode (required), --num_epochs<br><br>



Example calls:<br>
* Test Model Training<br>
`python language-module --mode=test --task=nl-ner --model_checkpoint=distilbert-base-uncased --dataset_name=cw1521/en-st-ner --model_name=nl-ner-10 --num_epochs=1`
<br>

* Normal Model Training<br>
`python language-module --mode=train --task=nl-ner --model_checkpoint=distilbert-base-uncased --dataset_name=cw1521/en-st-ner --model_name=nl-ner-10 --num_epochs=10`
<br>
