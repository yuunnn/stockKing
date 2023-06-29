# stockKing

## Step0 Make your environment
pip install -r requirements.txt

## Step 1 Initialize the database.
The `init.sh` script contains shell commands that initialize the SQLite3 database. It includes the company information and trading data. Please learn more from the `init.sh` and related .sql files.
```
cd database
sh init.sh
```

## Step2 Prepare the company info(a long-stand database, you may update it monthly or quarterly)
```
python get_all_symbol.py
python get_company_info.py
```

## Step3 Prepare the historical trading data
```
python get_history_market_price.py
```

## Step4 Prepare the daily trading data, do it daily when this project has started operating.
```
python get_daily_market_price.py
```

## Step 5 Prepare the training dataset, and train the model. 
Train as often as you see fit, but ideally train daily. It is recommended to update the model at least once a week.
```
python get_train_set.py
cd trainset; sh split_file.sh; cd ..
python train.py
```

## Step 6 Get you prediction
Please replace the model file path in `predict.py` with your own trained model file.
```
python get_predict_set.py
python predict.py
```
And you will find the recommended stock trading strategies in the `predictset`.