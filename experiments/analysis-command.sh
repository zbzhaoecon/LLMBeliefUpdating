python experiments/analysis.py
      experiments/data/exp1_stag_hunt_gpt-4.1-mini_20260305_184802.json
      experiments/data/exp1_pd_infinite_gpt-4.1-mini_20260305_200050.json
      experiments/data/exp1_beauty_contest_gpt-4.1-mini_20260305_201057.json
      experiments/data/exp1_auction_gpt-4.1-mini_20260305_201733.json
      experiments/data/exp1_ultimatum_gpt-4.1-mini_20260305_202456.json 2>&1


python experiments/analysis_exp2.py 2a
      experiments/data/exp2a_stag_hunt_X*_gpt-4.1-mini_20260305_*.json
      experiments/data/exp2a_stag_hunt_X*_gpt-4.1-mini_20260306_*.json
      experiments/data/exp2a_pd_X*_gpt-4.1-mini_20260305_*.json
      experiments/data/exp2a_pd_X*_gpt-4.1-mini_20260306_*.json 2>&1

python experiments/analysis_exp2.py 2b
      experiments/data/exp2b_*_gpt-4.1-mini_20260306_*.json 2>&1


python experiments/analysis_exp2.py 2c
      experiments/data/exp2c_*_gpt-4.1-mini_20260306_07*.json 2>&1
