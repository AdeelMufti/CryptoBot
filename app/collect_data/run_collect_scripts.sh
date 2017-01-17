#!/bin/bash
nohup python /home/ec2-user/cryptobot/app/collect_data/collect_btcc_books.py >> /dev/null 2>&1 &
nohup python /home/ec2-user/cryptobot/app/collect_data/collect_btcc_ticks.py >> /dev/null 2>&1 &
nohup python /home/ec2-user/cryptobot/app/collect_data/crawl_btcc_trades_history.py true >> /dev/null 2>&1 &
cd /home/ec2-user/cryptobot/app/
nohup python /home/ec2-user/cryptobot/app/create_live_features.py >> /dev/null 2>&1 &