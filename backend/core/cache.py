import time

_cache = {}
TTL = 60 * 60 * 6  # 6 hours

def cache_get(key):
    if key in _cache:
        val, expiry = _cache[key]
        if time.time() < expiry:
            return val
    return None

def cache_set(key, value):
    _cache[key] = (value, time.time() + TTL)