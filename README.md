# text-normalizer-flask

## Linux
### To install in Linux
#### Type in terminal
`source install.sh`

### To run in Linux
#### Type in terminal
Activate virutal environment
`source venv/bin/activate`


With test using twitter set
```
python3 app.py \
    --model_name=model_512 \
    --dataset_name=model_tagalog_b \
    --char_emb=True \
    --twitter=True
```


With test using test set
```
python3 app.py \
    --model_name=model_512 \
    --dataset_name=model_tagalog_b \
    --char_emb=True \
    --twitter=False
```


## Windows
### To install in Windows
#### Type in Command Prompt or Powershell
`install.bat`


### To run in Windows
#### Type in Command Prompt or Powershell
Activate virtual env
`venv/Scripts/activate`

With test using twitter set
`python app.py --model_name=model_512 --dataset_name=model_tagalog_b --char_emb=True --twitter=True`

With test using test set
`python app.py --model_name=model_512 --dataset_name=model_tagalog_b --char_emb=True --twitter=False`


#### Then access your localhost
`localhost:5000`
or
`127.0.0.1:5000`

