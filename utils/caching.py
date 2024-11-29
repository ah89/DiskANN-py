import redis
import pickle
from typing import Any


class RedisCache:
    """
    A caching utility using Redis as the backend.
    """

    def __init__(self, host="localhost", port=6379, db=0, ttl=None):
        """
        Initialize the RedisCache object.

        :param host: Redis server hostname or IP address.
        :param port: Redis server port.
        :param db: Redis database number.
        :param ttl: Time-to-live (TTL) for cached items in seconds. None means no expiration.
        """
        self.redis = redis.StrictRedis(host=host, port=port, db=db, decode_responses=False)
        self.ttl = ttl  # Default TTL for cached items

        # Test the connection
        try:
            self.redis.ping()
            print(f"Connected to Redis on {host}:{port} (db={db})")
        except redis.ConnectionError as e:
            raise Exception(f"Failed to connect to Redis: {e}")

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Store a value in the Redis cache.

        :param key: The cache key.
        :param value: The value to store (must be pickle-serializable).
        :param ttl: Optional TTL for this specific key in seconds. If None, use the default TTL.
        """
        # Serialize the value using pickle
        serialized_value = pickle.dumps(value)

        if ttl is None:
            ttl = self.ttl  # Use the default TTL if none is provided

        if ttl:
            self.redis.setex(key, ttl, serialized_value)
        else:
            self.redis.set(key, serialized_value)

    def get(self, key: str) -> Any:
        """
        Retrieve a value from the Redis cache.

        :param key: The cache key.
        :return: The cached value, or None if the key is not found.
        """
        serialized_value = self.redis.get(key)
        if serialized_value is None:
            return None
        # Deserialize the value using pickle
        return pickle.loads(serialized_value)

    def delete(self, key: str) -> None:
        """
        Delete a value from the Redis cache.

        :param key: The cache key.
        """
        self.redis.delete(key)

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the Redis cache.

        :param key: The cache key.
        :return: True if the key exists, False otherwise.
        """
        return self.redis.exists(key) > 0

    def clear(self) -> None:
        """
        Clear all keys in the Redis cache (for the current database).
        """
        self.redis.flushdb()

    def set_many(self, mapping: dict, ttl: int = None) -> None:
        """
        Store multiple key-value pairs in the Redis cache.

        :param mapping: A dictionary of key-value pairs to store.
        :param ttl: Optional TTL for these keys in seconds. If None, use the default TTL.
        """
        for key, value in mapping.items():
            self.set(key, value, ttl)

    def get_many(self, keys: list) -> dict:
        """
        Retrieve multiple values from the Redis cache.

        :param keys: A list of cache keys.
        :return: A dictionary of {key: value} pairs for found keys. Missing keys are excluded.
        """
        pipeline = self.redis.pipeline()
        for key in keys:
            pipeline.get(key)

        # Execute pipeline and deserialize results
        results = pipeline.execute()
        return {
            key: pickle.loads(value) for key, value in zip(keys, results) if value is not None
        }

    def set_with_expiry(self, key: str, value: Any, ttl: int) -> None:
        """
        Store a value in the Redis cache with a specific TTL.

        :param key: The cache key.
        :param value: The value to store (must be pickle-serializable).
        :param ttl: TTL for the key in seconds.
        """
        self.set(key, value, ttl)

    def expire(self, key: str, ttl: int) -> None:
        """
        Set a specific TTL for an existing key in the Redis cache.

        :param key: The cache key.
        :param ttl: The TTL in seconds.
        """
        self.redis.expire(key, ttl)