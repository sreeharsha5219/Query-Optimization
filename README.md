# lero
Implementation of Lero for DB Design (CS 6360) project.

# Steps to run the code

### Install postgres (with lero patch)
1. `unzip postgresql-13.1.zip` and `cd postgresql-13.1`
2. install postgres
```
./configure
sudo make
sudo make install
```
3. create data path for postgres
```
mkdir ~/Desktop/pgsql/data
```
4. start postgres server
```
/usr/local/pgsql/bin/pg_ctl -D ~/Desktop/pgsql/data -l logfile start
```
5. create `postgres` user and give root access
```
/usr/local/pgsql/bin/psql test
CREATE ROLE postgres;
ALTER ROLE postgres WITH PASSWORD 'postgres';
ALTER ROLE postgres SUPERUSER;
```
6. load stats database
```
/usr/local/pgsql/bin/psql -d stats -f stats_db.sql
```

### Lero Training
Update the `PG_DB_PATH` entry in `lero.conf` to point to the data path of postgres, the one created in earlier steps.

1. Start the connector server
```
python postgres-connector.py   
```
2. run training script
```
python train_lero.py --query_path data/train/stats.txt --test_query_path data/test/stats.txt --query_num_per_chunk 20 --output_query_latency_file lero_stats.log --model_prefix stats_model --topK 3
```
All the test results will get auto logged onto the current directory.

3. Run queries with naive PG optimiser
```
python run_pg.py --query_path ./data/train/stats.txt --output_query_latency_file pg_stats_train.log
python run_pg.py --query_path ./data/test/stats.txt --output_query_latency_file pg_stats_test.log
```
This will generate all the necessary logs to generate the experiment graphs

Run the `visualization.ipynb` notebook to generate the train and test results when compared with the results of
postgres naive optimiser.

# Acknowledgements
TreeConvolution implementation has been taken from RyanMarcus's [repo](https://github.com/RyanMarcus/TreeConvolution). 
