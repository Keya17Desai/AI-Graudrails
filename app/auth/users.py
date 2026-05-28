"""
In-memory user store with bcrypt-hashed passwords.
In production this would be a PostgreSQL table — but for learning,
keeping it in-memory makes the auth logic clear without DB plumbing.
"""
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# Seeded users: username → {hashed_password, role}
# Roles: employee, hr, finance, marketing, c_level
USERS: dict[str, dict] = {
    "alice": {
        "hashed_password": hash_password("alice123"),
        "role": "employee",
        "full_name": "Alice Chen",
    },
    "bob": {
        "hashed_password": hash_password("bob123"),
        "role": "hr",
        "full_name": "Bob Smith",
    },
    "carol": {
        "hashed_password": hash_password("carol123"),
        "role": "finance",
        "full_name": "Carol Rivera",
    },
    "dave": {
        "hashed_password": hash_password("dave123"),
        "role": "marketing",
        "full_name": "Dave Park",
    },
    "eve": {
        "hashed_password": hash_password("eve123"),
        "role": "c_level",
        "full_name": "Eve Johnson",
    },
}


def get_user(username: str) -> dict | None:
    return USERS.get(username)


def authenticate(username: str, password: str) -> dict | None:
    """Return user dict if credentials are valid, else None."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
