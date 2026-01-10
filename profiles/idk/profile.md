sudoku_ga on  main [!?] is 󰏗 v0.1.0 via  v3.13.9 took 6s 
❯ uv run py-spy top -- python src/main.py


Collecting samples from 'python src/main.py' (python v3.12.12)
Total Samples 600
GIL: 100.00%, Active: 100.00%, Threads: 1

  %Own   %Total  OwnTime  TotalTimeFunction(filename)
  63.00%  63.00%    3.70s     3.70s   calculate_population_fitness (ga.py)
  4.00%  13.00%   0.570s    0.970s   sample (random.py)
  5.00%  11.00%   0.440s    0.640s   crossover (ga.py)
 11.00%  11.00%   0.420s    0.420s   _randbelow_with_getrandbits (random.py)
  9.00%  17.00%   0.340s     1.11s   get_parents_from_tournament (ga.py)
  2.00%   7.00%   0.140s    0.340s   mutate (ga.py)
  1.00%   2.00%   0.100s    0.160s   __instancecheck__ (<frozen abc>)
  1.00%   1.00%   0.060s    0.070s   deepcopy (copy.py)
  0.00%   1.00%   0.050s    0.060s   __subclasscheck__ (<frozen abc>)
  0.00% 100.00%   0.050s     5.99s   run_evolution (ga.py)
  0.00%  35.00%   0.030s     2.14s   evolve_population (ga.py)
  1.00%   2.00%   0.020s    0.110s   make_initial_population (ga.py)
  2.00%   6.00%   0.020s    0.190s   randrange (random.py)
  0.00%   0.00%   0.020s    0.020s   <lambda> (ga.py)
  0.00%   6.00%   0.010s    0.200s   randint (random.py)
  0.00%   1.00%   0.010s    0.070s   _deepcopy_list (copy.py)
  1.00%   1.00%   0.010s    0.010s   __subclasshook__ (<frozen _collections_abc>)
  0.00%   0.00%   0.010s    0.020s   shuffle (random.py)
  0.00% 100.00%   0.000s     6.00s   run_once (main.py)
  0.00% 100.00%   0.000s     6.00s   <module> (main.py)
