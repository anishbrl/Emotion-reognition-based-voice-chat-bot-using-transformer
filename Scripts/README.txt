This folder contains all pretrained models used in the project.
Files description:

1. get_emotion.py loads pretrained emotion models from emotion/
2. generator.py builds and loads pretrained transformer  model from model_checkpoint/ and generates replies.
3. history.py saves and loads history conversations.
4. intent.py loads pretrained intent models from intent/
5. intent+emotion+context.tf is custom subword tokenizer used.
6. test2.py complains all the files and connects to API.py 
