ECHO OFF

FOR /L %%i IN (11,1,1000000) DO (
    python exp_MNIST_prime.py --run %%i
)

PAUSE
