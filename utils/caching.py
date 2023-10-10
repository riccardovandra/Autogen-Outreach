import requests
import json
from functools import wraps

class CacheManager:
    def __init__(self, cache_file):
        self.cache_file = cache_file

    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_cache(self, cache_data):
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=4)

    def cache(self, key_fn):
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                cache_data = self._load_cache()
                cache_key = key_fn(*args, **kwargs)
                for entry in cache_data:
                    if entry['cache_key'] == cache_key:
                        print('Fetched data from Local Cache')
                        return entry['response']
                print('Fetching new data... (updating local cache)')
                response = func(*args, **kwargs)
                new_data = {
                    'cache_key': cache_key,
                    'response': response
                }
                cache_data.append(new_data)
                self._save_cache(cache_data)
                return response
            return wrapped
        return decorator

cache_manager = CacheManager('json_cache.json')