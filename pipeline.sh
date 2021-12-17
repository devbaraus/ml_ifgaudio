# screen -L -Logfile mfcc_log python3 representation_svm.py -f mfcc -c 10 18 26 34 40 -s 4 3 2 -a 10
# screen -L -Logfile lpc_log -S lpc_test python3 representation_svm.py -f lpc -c 512 1024 2048 -s 4 3 2 1 -a 10
python3 representation_svm.py -f fft -c 256 512 1024 2048 -s 4 3 2 1 -a 10
python3 representation_svm.py -f stft -c 256 512 1024 2048 -s 3 2 1 -a 10
python3 representation_svm.py -f lpc -c 256 512 1024 2048 -s 4 3 2 1 -a 10
# screen -L -Logfile stft_log -S stft_test python3 representation_svm.py -f stft -c 512 1024 2048 -s 4 3 2 1 -a 10