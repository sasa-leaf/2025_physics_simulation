import time
import taichi as ti
ti.init(arch=ti.cpu) # Taichiの初期化

@ti.func # Taichi関数
def is_prime(n: int):
  result = True
  for k in range(2,int(n**0.5)+1):
    if n % k == 0:
      result = False
      break
  return result

@ti.kernel # Taichiカーネル
def count_primes(n: int) -> int:
  count = 0
  for k in range(2,n):
    if is_prime(k):
      count += 1
  return count

comp_start = time.perf_counter()

print(count_primes(1000000))

comp_end = time.perf_counter()

print('comptime:',comp_end-comp_start,'sec')
