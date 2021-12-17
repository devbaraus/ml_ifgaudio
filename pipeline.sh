screen -L -Logfile mfcc_log python3 representation_svm.py -f mfcc -c 10 18 26 34 40 -s 4 3 2 -a 10
screen -L -Logfile mfcc_log python3 representation_svm.py -f mfcc -c 10 18 26 34 40
screen -L -Logfile lpc_log python3 representation_svm.py -f lpc -c 512 1024 2048
screen -L -Logfile stft_log python3 representation_svm.py -f stft -c 512 1024 2048