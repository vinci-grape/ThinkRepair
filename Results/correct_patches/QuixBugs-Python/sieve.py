def sieve(limit):
    primes = []
    for n in range(2, limit + 1):
        if all(n % p > 0 for p in primes):
            primes.append(n)
    return primes