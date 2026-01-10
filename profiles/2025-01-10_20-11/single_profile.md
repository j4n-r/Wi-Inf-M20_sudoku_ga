sudoku_ga on  main [✘!?] is 󰏗 v0.1.0 via  v3.13.9 took 4s 
❯ uv run py-spy  top --subprocesses -- python src/main.py


Collecting samples from 'python src/main.py' and subprocesses
Total Samples 300
GIL: 100.00%, Active: 100.00%, Threads: 1, Processes 1

  %Own   %Total  OwnTime  TotalTime  Function (filename)                                                                                           46.00%  46.00%    1.22s     1.22s   calculate_fitness (ga.py)
 28.00%  28.00%   0.600s    0.600s   _randbelow_with_getrandbits (random.py)
  4.00%  14.00%   0.230s    0.540s   sample (random.py)
  5.00%   5.00%   0.210s    0.230s   __instancecheck__ (<frozen abc>)
  0.00%   0.00%   0.150s    0.270s   deepcopy (copy.py)
  8.00%  32.00%   0.140s    0.640s   crossover (ga.py)
  0.00%   0.00%   0.120s    0.270s   _deepcopy_list (copy.py)
  2.00%  52.00%   0.080s     1.37s   evolve_population (ga.py)
  1.00%   7.00%   0.060s    0.340s   mutate (ga.py)
  3.00%  11.00%   0.050s    0.310s   get_parents_from_tournament (ga.py)
  2.00% 100.00%   0.040s     2.63s   run_evolution (ga.py)
  1.00%  24.00%   0.030s    0.500s   randrange (random.py)
  0.00%   0.00%   0.020s    0.020s   _compile_bytecode (<frozen importlib._bootstrap_external>)
  0.00%   0.00%   0.020s    0.070s   shuffle (random.py)
  0.00%   0.00%   0.020s    0.020s   __subclasscheck__ (<frozen abc>)
  0.00%   0.00%   0.010s    0.350s   make_initial_population (ga.py)
  0.00%   0.00%   0.000s    0.020s   _handle_fromlist (<frozen importlib._bootstrap>)
  0.00%   0.00%   0.000s    0.020s   _find_and_load_unlocked (<frozen importlib._bootstrap>)
  0.00%   0.00%   0.000s    0.020s   get_code (<frozen importlib._bootstrap_external>)
  0.00% 100.00%   0.000s     3.00s   <module> (main.py)
  0.00%   0.00%   0.000s    0.020s   <module> (ga.py)
  0.00%   0.00%   0.000s    0.020s   _call_with_frames_removed (<frozen importlib._bootstrap>)
  0.00%   0.00%   0.000s    0.020s   _load_unlocked (<frozen importlib._bootstrap>)
  0.00%  24.00%   0.000s    0.500s   randint (random.py)
  0.00%   0.00%   0.000s    0.020s   _find_and_load (<frozen importlib._bootstrap>)
  0.00%   0.00%   0.000s    0.020s   <module> (multiprocessing/connection.py)
  0.00%   0.00%   0.000s    0.020s   <module> (concurrent/futures/process.py)
  0.00%   0.00%   0.000s    0.020s   <module> (multiprocessing/util.py)
  0.00% 100.00%   0.000s     2.98s   run_once (main.py)
  0.00%   0.00%   0.000s    0.020s   exec_module (<frozen importlib._bootstrap_external>)
  0.00%   0.00%   0.000s    0.020s   __getattr__ (concurrent/futures/__init__.py)

































