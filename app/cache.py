from pathlib import Path
import typing as t
import joblib
from logging import getLogger

logger = getLogger(__name__)

F = t.TypeVar("F", bound=t.Callable[..., t.Any])


class Cache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def __call__(self, key: str, f: F) -> F:
        path = self.cache_dir.joinpath(key)

        def execute(*args: t.Any, **kwargs: t.Any) -> t.Any:
            logger.info(f"execute {key}")
            res = f(*args, **kwargs)
            joblib.dump(res, path)
            return res

        def skip(*args: t.Any, **kwargs: t.Any) -> t.Any:
            logger.info(f"skip {key}")
            res = joblib.load(path)
            return res

        if path.exists():
            return t.cast(F, skip)
        else:
            return t.cast(F, execute)
