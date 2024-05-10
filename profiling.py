import pstats
from pstats import SortKey

p = pstats.Stats('profile.prof')
# p = pstats.Stats('profile_vectorized.prof')
# p.strip_dirs().sort_stats(-1).print_stats(20)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)
# p.sort_stats(SortKey.CUMULATIVE).print_stats(30)