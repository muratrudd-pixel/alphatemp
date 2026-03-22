#!/bin/bash
cd ~/Projects/alphatemp/alphatemp
source .env
exec ~/Projects/alphatemp/alphatemp/venv/bin/python main.py --dashboard
