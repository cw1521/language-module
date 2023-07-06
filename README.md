# Language-Module
## Requirements
Python 3.10 required<br><br>
To install the required packages:<br>
pip install transformers datasets numpy seqeval sacrebleu<br><br>

## Usage
This program trains transformers using the HuggingFace API<br><br>
st-en: Translation task<br>
en-st: Translation task<br>
ner-st: Translation task<br>
nl-ner: Named Entity Recognition task<br><br>
decrease batch_size for smaller machines<br><br>


### CLI
Use -h or --help for help<br><br>
num_epochs=10 (default) <br>
mode=test (runs a test with 10% of data and smaller batch size)
<br><br>
Argument list:<br>
* --task (required), --model_checkpoint (required), --dataset_name (required), 
--model_name (required), --mode, --num_epochs<br><br>



Example calls:<br>
* Test<br>
`python language-module --task=nl-ner --model_checkpoint=distilbert-base-uncased --dataset_name=cw1521/en-st-ner --model_name=nl-ner-10 --num_epochs=1 --mode=test`
<br>

* Normal<br>
`python language-module --task=nl-ner --model_checkpoint=distilbert-base-uncased --dataset_name=cw1521/en-st-ner --model_name=nl-ner-10 --num_epochs=10`
<br>
