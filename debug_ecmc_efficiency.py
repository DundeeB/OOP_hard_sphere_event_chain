from run_functions import *

initial_time = time.time()
realizations, displacements = run_square(0.8, 90, 90, 0.79, record_displacements=True, iterations=2000)
# iterations per minute are of order 10
final_time = time.time()
print('Displacements per sec = ' + str(displacements[-1] / (final_time - initial_time)))
