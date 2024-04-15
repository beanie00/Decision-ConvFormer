# Decision ConvFormer (DC)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Breakout' --batch_size 128 --token_mixer 'conv'
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Qbert' --batch_size 128 --token_mixer 'conv'
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Pong' --batch_size 512 --token_mixer 'conv'
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Seaquest' --batch_size 128 --token_mixer 'conv'
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Asterix' --batch_size 128 --token_mixer 'conv'
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Frostbite' --batch_size 128 --token_mixer 'conv'
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Assault' --batch_size 128 --token_mixer 'conv'
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 8 --game 'Gopher' --batch_size 128 --token_mixer 'conv'
done


# Decision Transformer (DT)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --game 'Breakout' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --game 'Qbert' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 50 --game 'Pong' --batch_size 512
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --game 'Seaquest' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --game 'Asterix' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --game 'Frostbite' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --game 'Assault' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --game 'Gopher' --batch_size 128
done
