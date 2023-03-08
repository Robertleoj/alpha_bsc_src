#!/bin/bash

ssh gimli@smallvoice.ru.is -t "cd ~/AlphaBSc/alpha_bsc_src && tar -czvf data.tar.gz ./db/db.db ./models"
scp gimli@smallvoice.ru.is:/home/gimli/AlphaBSc/data.tar.gz ./