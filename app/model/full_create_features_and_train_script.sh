#!/bin/bash

echo "---------------Starting features create-----------------"
python -W ignore -u features_parallel.py 0 0 ../data/data.pkl

# https://aws.amazon.com/ec2/pricing/on-demand/ c4.8xlarge = 36 processors, 60gb ram = $1.675/hr
# With 3938164 records, each instance takes memory: virt=9.8gb, res=7.3g =~ 55gb used for 7 in free -g =~ 7.8gb/instance
echo "---------------Starting training-----------------"
nohup python strategy.py ../data/data.tsv 99 5 0.01 true >> ../data/5.txt &
nohup python strategy.py ../data/data.tsv 99 10 0.01 true >> ../data/10.txt &
nohup python strategy.py ../data/data.tsv 99 15 0.01 true >> ../data/15.txt &
nohup python strategy.py ../data/data.tsv 99 20 0.01 true >> ../data/20.txt &
nohup python strategy.py ../data/data.tsv 99 25 0.01 true >> ../data/25.txt &
nohup python strategy.py ../data/data.tsv 99 30 0.01 true >> ../data/30.txt &
nohup python strategy.py ../data/data.tsv 99 35 0.01 true >> ../data/35.txt &
nohup python strategy.py ../data/data.tsv 99 40 0.01 true >> ../data/40.txt &
nohup python strategy.py ../data/data.tsv 99 45 0.01 true >> ../data/45.txt &

while pgrep &>/dev/null -f python; do sleep 60; done; shutdown -h now